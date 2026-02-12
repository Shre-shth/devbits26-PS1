from piper import PiperVoice
import pyaudio
import queue
import threading
import os
import numpy as np
import time
import struct
import socket
import scipy.signal


class Synthesizer:
    def __init__(self, voicebot=None, model_path="en_US-ryan-medium.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Piper model not found at {model_path}")

        self.voice = PiperVoice.load(model_path)
        self.voicebot = voicebot

        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.current_energy = 0.0
        self.is_synthesizing = False

        # ================= RTP CONFIG =================
        # Dynamic address from VoiceBot (set when packet is received)
        self.asterisk_ip = None
        self.asterisk_port = None

        self.rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.sequence = 0
        self.timestamp = 0
        self.ssrc = 12345
        # ==============================================

        # Local playback (for debugging)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=22050,
            output=True
        )

        self.abort_event = threading.Event()

        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True
        )
        self.playback_thread.start()

    # ==================================================
    # RTP SEND FUNCTION
    # ==================================================

    def send_rtp(self, pcm_data):
        # Convert to big-endian (Asterisk requires big-endian)
        audio_np = np.frombuffer(pcm_data, dtype=np.int16)
        be_audio = audio_np.byteswap().tobytes()

        # RTP header (12 bytes)
        rtp_header = struct.pack(
            "!BBHII",
            0x80,   # RTP version 2
            10,     # Payload type 10 (L16)
            self.sequence,
            self.timestamp,
            self.ssrc
        )

        packet = rtp_header + be_audio

        try:
            # USE THE BOUND SOCKET (Port 4000) from VoiceBot
            # This ensures Symmetric RTP (Source Port 4000 -> Dest Port 4000)
            if self.voicebot and hasattr(self.voicebot, 'remote_addr') and self.voicebot.remote_addr:
                target_ip, target_port = self.voicebot.remote_addr
                self.voicebot.rtp_socket.sendto(packet, (target_ip, target_port))
            else:
                 # Fallback (Should not happen in main app if connected)
                 pass
                 
        except Exception as e:
            print(f"[RTP] Send Error: {e}")

        self.sequence = (self.sequence + 1) % 65536
        self.timestamp += len(pcm_data) // 2
    # ==================================================
    # PLAYBACK WORKER (SENDS RTP FRAMES)
    # ==================================================

    def _playback_worker(self):
        CHUNK_SIZE = 640  # 20ms @16kHz

        while True:
            audio_data = self.audio_queue.get()

            if audio_data is None:
                self.audio_queue.task_done()
                break

            try:
                for i in range(0, len(audio_data), CHUNK_SIZE):
                    if self.abort_event.is_set():
                        break

                    chunk = audio_data[i:i + CHUNK_SIZE]

                    if len(chunk) < CHUNK_SIZE:
                        break

                    # Energy calculation (for echo gate)
                    audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                    if len(audio_np) > 0:
                        self.current_energy = np.sqrt(np.mean(audio_np ** 2))

                    self.is_playing = True

                    # 🔥 SEND TO ASTERISK
                    self.send_rtp(chunk)

                    # 20ms pacing
                    time.sleep(0.02)

            except Exception as e:
                print(f"[TTS] Playback Error: {e}")
            finally:
                self.current_energy = 0.0
                self.audio_queue.task_done()

    # ==================================================
    # CONTROL FUNCTIONS
    # ==================================================

    def stop(self):
        self.abort_event.set()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

    def reset(self):
        self.abort_event.clear()

    # ==================================================
    # TTS ENTRY POINT
    # ==================================================

    def synthesize_text(self, text):
        self.is_playing = True
        self._synthesize_enqueue(text)

    def _synthesize_enqueue(self, text):
        if self.abort_event.is_set():
            return

        try:
            self.is_synthesizing = True
            print(f"[TTS] Synthesizing: '{text}'")

            stream = self.voice.synthesize(text)

            for audio_chunk in stream:
                if self.abort_event.is_set():
                    return

                audio_bytes = audio_chunk.audio_int16_bytes
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

                if len(audio_np) > 0:
                    # 🔥 RESAMPLE 22050 → 16000
                    resampled = scipy.signal.resample_poly(audio_np, 16000, 22050)
                    pcm_bytes = resampled.astype(np.int16).tobytes()

                    self.audio_queue.put(pcm_bytes)

            print("[TTS] Audio enqueued")

        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.is_synthesizing = False

    # ==================================================
    # STATUS HELPERS
    # ==================================================

    @property
    def is_active(self):
        return (self.audio_queue.unfinished_tasks > 0) or self.is_synthesizing

    def wait_until_done(self):
        self.audio_queue.join()
        self.is_playing = False

    def close(self):
        self.audio_queue.put(None)
        self.playback_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()