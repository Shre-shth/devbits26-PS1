from piper import PiperVoice
import queue
import threading
import os
import numpy as np
import time
import struct
import socket
import scipy.signal


class Synthesizer:
    def __init__(self, voicebot=None, model_path="en_US-amy-low.onnx"):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Piper model not found at {model_path}")

        self.voice = PiperVoice.load(model_path)
        self.voicebot = voicebot
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.current_energy = 0.0
        self.is_synthesizing = False
        self.first_packet_of_turn = True

        # Detect model sample rate
        try:
            if hasattr(self.voice.config, 'sample_rate'):
                self.model_rate = self.voice.config.sample_rate
                print(f"[Synthesizer] Detected model sample rate: {self.model_rate} Hz")
            else:
                self.model_rate = 16000
                print(f"[Synthesizer] Defaulting model sample rate to: {self.model_rate} Hz")
        except:
            self.model_rate = 16000
            print(f"[Synthesizer] Error reading config, defaulting to: {self.model_rate} Hz")
        
        print(f"Model sample rate: {self.model_rate}")

        # RTP CONFIG
        self.asterisk_ip = "127.0.0.1"
        self.asterisk_port = 4000

        self.rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        if self.voicebot is None:
            try:
                self.rtp_socket.bind(("0.0.0.0", 4000))
                print("[RTP] Listening on 0.0.0.0:4000 (Standalone Mode)")
            except Exception as e:
                print(f"[RTP] Bind Error: {e}")
        else:
            print("[RTP] Running in Integrated Mode")

        self.sequence = 0
        self.timestamp = 0
        self.ssrc = 12345
        self.output_rate = 16000
        self.next_packet_time = 0.0

        self.abort_event = threading.Event()

        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True
        )
        self.playback_thread.start()

    # ==================================================
    # RTP SEND FUNCTION
    # ==================================================

    def send_rtp(self, pcm_data, marker=False):
        payload_type = 11  # SLIN16 STATIC
        # print(">>> PAYLOAD TYPE USED:", payload_type) # Fixed UnboundLocalError
        audio_np = np.frombuffer(pcm_data, dtype=np.int16)
        be_audio = audio_np.byteswap().tobytes()

        byte1 = payload_type | (0x80 if marker else 0x00)

        rtp_header = struct.pack(
            "!BBHII",
            0x80,
            byte1,
            self.sequence,
            self.timestamp,
            self.ssrc
        )

        packet = rtp_header + be_audio

        if self.voicebot and self.voicebot.remote_addr:
            self.voicebot.rtp_socket.sendto(packet, self.voicebot.remote_addr)
        else:
            self.rtp_socket.sendto(packet, (self.asterisk_ip, self.asterisk_port))

        self.sequence = (self.sequence + 1) % 65536
        self.timestamp += len(pcm_data) // 2

    # ==================================================
    # FIXED PLAYBACK WORKER
    # ==================================================

    def _playback_worker(self):

        FRAME_BYTES = 320  # 20ms @ 8kHz (160 samples)
        FRAME_DURATION = 0.02  # 20ms

        while True:
            audio_data = self.audio_queue.get()

            if audio_data is None:
                self.is_playing = False
                self.audio_queue.task_done()
                break

            try:
                now = time.time()
                if now > self.next_packet_time + 0.1:
                    self.next_packet_time = now

                buffer = audio_data

                while len(buffer) >= FRAME_BYTES:

                    if self.abort_event.is_set():
                        break

                    chunk = buffer[:FRAME_BYTES]
                    buffer = buffer[FRAME_BYTES:]

                    self.is_playing = True

                    marker = self.first_packet_of_turn
                    if marker:
                        print("[DEBUG] Starting new talkspurt (Marker=True)")
                        self.first_packet_of_turn = False

                    self.send_rtp(chunk, marker=marker)

                    self.next_packet_time += FRAME_DURATION
                    sleep_time = self.next_packet_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except Exception as e:
                print(f"[Playback Error] {e}")

            finally:
                self.current_energy = 0.0
                self.is_playing = False
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
        self.first_packet_of_turn = True

    # ==================================================
    # TTS ENTRY
    # ==================================================

    def synthesize_text(self, text):

        self.is_playing = True

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
                    # ALWAYS output 8000 Hz
                    if self.model_rate != 8000:
                        resampled = scipy.signal.resample_poly(
                            audio_np, 8000, self.model_rate
                        )
                        pcm_bytes = resampled.astype(np.int16).tobytes()
                    else:
                        pcm_bytes = audio_bytes

                    self.audio_queue.put(pcm_bytes)

            print("[TTS] Audio enqueued")

        except Exception as e:
            print(f"[TTS Error] {e}")

        finally:
            self.is_synthesizing = False

    @property
    def is_active(self):
        return (self.audio_queue.unfinished_tasks > 0) or self.is_synthesizing or self.is_playing

    def wait_until_done(self):
        self.audio_queue.join()
        self.is_playing = False

    def close(self):
        self.audio_queue.put(None)
        self.playback_thread.join()
