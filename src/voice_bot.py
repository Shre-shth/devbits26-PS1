import time
import socket
import threading
import json
import numpy as np
import scipy.signal
import re
from config import RATE, CHUNK_SIZE, API_KEY, VAD_MODEL_PATH, VOICE_MODEL
from utils import CallState, AudioPreprocessor
from vad import VADIterator
from transcriber import Transcriber
from brain import Brain
from synthesizer import Synthesizer

class VoiceBot:
    def __init__(self):
        print("Initializing modules...")
        self.vad = VADIterator(model_path=VAD_MODEL_PATH, threshold=0.5, min_silence_duration_ms=500, sampling_rate=RATE)
        self.transcriber = Transcriber()
        self.brain = Brain(API_KEY)
        self.synthesizer = Synthesizer(self, model_path=VOICE_MODEL)
        self.audio_processor = AudioPreprocessor(RATE, gain=3.0)

        self.state = CallState.LISTENING
        self.buffer = bytearray()
        self.last_state_change = time.time()

        self.barge_in_chunks = 0
        self.BARGE_IN_CHUNKS_THRESHOLD = 4 # ~125ms (Faster interruption)
        self.pre_barge_buffer = bytearray()
        
        self.ECHO_GATE_THRESHOLD = 2000 
        self.STATE_TRANSITION_GRACE_MS = 620

        self.rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtp_socket.bind(("127.0.0.1", 4000))

        self.remote_addr = None
        self.asterisk_signaled_port = None
        self.incoming_buffer = bytearray()
        self.last_audio_active_time = time.time()

        self.should_terminate = False
        self.is_generating = False
        self.call_connected = threading.Event()

    def run(self):
        print("\n--- Voice Bot Started (Local RTP Mode) ---")
        print("Listening on 127.0.0.1:4000...")
        print(f"State: {self.state.name}")

        print("Waiting for ARI StasisStart event...")
        self.call_connected.wait()
        print("ARI Connected! Waiting for RTP packet to establish address...")

        try:
            packet, addr = self.rtp_socket.recvfrom(2048)
            if hasattr(self, "asterisk_signaled_port") and self.asterisk_signaled_port:
                self.remote_addr = (addr[0], self.asterisk_signaled_port)
                print(f"Connection established from {addr} -> Sending to {self.remote_addr} (ARI Port)")
            else:
                self.remote_addr = addr
                print(f"Connection established from {addr} (Symmetric Port)")
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"Socket error: {e}")
            return

        self.rtp_socket.setblocking(False)

        time.sleep(0.5)

        call_type = getattr(self, "call_type", "inbound")
        chan_name = getattr(self, "channel_name", "")

        if call_type == "outbound" or "6002" in chan_name:
            print(f"[DEBUG] Generating OUTBOUND greeting (type={call_type}, chan={chan_name})")
            
            outbound_context = (
                "Current Situation: You represent Shreshth Enterprises. You are making a follow-up call to this user. "
                "You spoke 4 days ago about their interest in buying a flat. "
                "Your Goal: Ask if they have made a decision or if they would like you to send a detailed report of all available flats in their area. "
                "User's Name: Jon Snow  "
                "Ask for the method to share detailed report eg via email or whatsapp and his further plans "
                "Do not ask for their name again."
                "If the user disagree to take the option, ask for the reason and try to resolve the issue, but dont force the user."
            )
            self.brain.update_system_instruction(outbound_context)
            
            greeting_text = (
                "Hello Jon Snow. I am calling you on behalf of Shreshth Enterprises. We had a call 4 days ago regarding your interest in buying a flat. "
                "Have you decided on how to proceed?"
            )
        else:
            print(f"[DEBUG] Generating INBOUND greeting (type={call_type}, chan={chan_name})")
            greeting_text = "Hello! Welcome to Shreshth Enterprises. How can I assist you today?"

        print(f"Bot: {greeting_text}")
        
        self.brain.history.append({"role": "model", "parts": [greeting_text]})
        
        self.synthesizer.synthesize_text(greeting_text)
        self.set_state(CallState.SPEAKING)

        try:
            while True:
                try:
                    packet, addr = self.rtp_socket.recvfrom(2048)

                    if self.remote_addr is None:
                        if hasattr(self, "asterisk_signaled_port") and self.asterisk_signaled_port:
                            self.remote_addr = (addr[0], self.asterisk_signaled_port)
                            print(f"[RTP] Using ARI signaled port for destination: {self.remote_addr}")
                        else:
                            self.remote_addr = addr
                            print(f"[RTP] Using symmetric destination port: {self.remote_addr}")

                    raw_rtp_audio = packet[12:]

                    if len(raw_rtp_audio) > 0:
                        audio_8k = np.frombuffer(raw_rtp_audio, dtype='>i2').astype(np.int16)

                        audio_16k = scipy.signal.resample_poly(audio_8k, 2, 1).astype(np.int16)

                        self.incoming_buffer.extend(audio_16k.tobytes())
                    else:
                        continue

                except BlockingIOError:
                    continue
                except IOError as e:
                    print(f"Audio Error: {e}")
                    continue

                while len(self.incoming_buffer) >= CHUNK_SIZE * 2: 
                    chunk = self.incoming_buffer[:CHUNK_SIZE * 2]
                    self.incoming_buffer = self.incoming_buffer[CHUNK_SIZE * 2:]
                    self.process_audio(chunk)

                if self.state == CallState.SPEAKING:
                    self.check_playback_status()

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.shutdown()

    def process_audio(self, audio_chunk):
        # --- 1. Spectral Filtering (Improve SNR) ---
        audio_int16 = self.audio_processor.process(audio_chunk)
        processed_chunk = audio_int16.tobytes()

        # --- 2. VAD Logic (Run FIRST) ---
        vad_event = self.vad(audio_int16)
        prob = vad_event.get('prob', 0.0)

        # --- 3. Echo Gate (Reference Logic) ---
        is_barge_in_candidate = False

        if self.state == CallState.SPEAKING:
            # Grace period check
            if (time.time() - self.last_state_change) * 1000 < self.STATE_TRANSITION_GRACE_MS:
                return

            energy = np.abs(audio_int16).mean()
            
            output_energy = getattr(self.synthesizer, "current_energy", 0.0)
            dynamic_threshold = max(self.ECHO_GATE_THRESHOLD, output_energy * 0.8) 
            
            if energy > dynamic_threshold:
                is_barge_in_candidate = True
            else:
                if energy > 2000:
                    print(f"\r[Echo Gate] Ignored. Energy: {energy:.1f} < {dynamic_threshold:.1f} (Out: {output_energy:.1f})      ", end="", flush=True)
                return

        # --- 4. Main State Handling ---
        if self.state == CallState.LISTENING:
            self.handle_listening(processed_chunk, vad_event)
        
        elif self.state == CallState.SPEAKING or self.state == CallState.THINKING:
            if is_barge_in_candidate:
                self.handle_barge_in(processed_chunk, prob)

    def handle_listening(self, audio_chunk, vad_event):
        if vad_event.get('start') is not None:
             print("\n[Speech Detected]")
             
        if vad_event.get('end') is not None:
             print(f"\n[End of Turn] Buffer: {len(self.buffer)} bytes")
             
             self.set_state(CallState.THINKING)
             full_audio = bytes(self.buffer)
             self.buffer = bytearray()
             self.vad.reset_states() 
             
             self.process_turn(full_audio)
             return

        if self.vad.triggered:
             self.buffer.extend(audio_chunk)

    def handle_barge_in(self, audio_chunk, prob):
        if prob > self.vad.threshold:
            self.barge_in_chunks += 1
            self.pre_barge_buffer.extend(audio_chunk) 
            
            if self.barge_in_chunks >= self.BARGE_IN_CHUNKS_THRESHOLD:
                print("\n[!] BARGE-IN DETECTED - (Duration > 125ms)")
                
                try:
                    self.synthesizer.stop()
                except Exception as e:
                    print(f"Barge-In Stop Error: {e}")
                
                temp_audio_capture = self.pre_barge_buffer[:] 

                self.set_state(CallState.LISTENING)
                
                self.buffer.extend(temp_audio_capture)

                self.pre_barge_buffer = bytearray() 
                
                self.vad.triggered = True 
        else:
             self.barge_in_chunks = 0
             self.pre_barge_buffer = bytearray()

    def set_state(self, new_state):
        self.state = new_state
        self.last_state_change = time.time()

        if new_state == CallState.LISTENING:
            print("\n[STATE] Listening...")
            self.vad.reset_states()
            self.buffer = bytearray()
            self.pre_barge_buffer = bytearray()
            self.barge_in_chunks = 0

    def process_turn(self, audio_data):
        self.synthesizer.reset()

        text = self.transcriber.transcribe(audio_data)
        print(f"User: {text}")

        if not text.strip():
            self.set_state(CallState.LISTENING)
            return

        threading.Thread(
            target=self._generation_worker,
            args=(text,),
            daemon=True
        ).start()

    def _generation_worker(self, text):
        self.is_generating = True
        sentence_buffer = ""

        try:
            stream = self.brain.generate_response_stream(text)

            for chunk in stream:
                if self.state == CallState.LISTENING:
                    self.is_generating = False
                    return

                sentence_buffer += chunk

                parts = re.split(r'(?<=[.?!;])\s+', sentence_buffer)

                if len(parts) == 1 and len(sentence_buffer) > 40:
                    sub_parts = re.split(r'(?<=[,:\-])\s+', sentence_buffer)
                    if len(sub_parts) > 1:
                        parts = sub_parts

                if len(parts) > 1:
                    for i in range(len(parts) - 1):
                        to_speak = parts[i].strip()
                        if to_speak:
                            print(f"[DEBUG] Requesting synthesis for: '{to_speak}'")
                            if self.state in (CallState.THINKING, CallState.SPEAKING):
                                if self.state == CallState.THINKING:
                                    self.set_state(CallState.SPEAKING)
                                self.synthesizer.synthesize_text(to_speak)

                    sentence_buffer = parts[-1]

            if sentence_buffer.strip() and self.state != CallState.LISTENING:
                print(f"[DEBUG] Requesting residual synthesis for: '{sentence_buffer.strip()}'")
                self.synthesizer.synthesize_text(sentence_buffer.strip())

        except Exception as e:
            print(f"Generation Error: {e}")
            self.set_state(CallState.LISTENING)
        finally:
            self.is_generating = False

    def check_playback_status(self):
        if self.synthesizer.is_active or self.is_generating:
            self.last_audio_active_time = time.time()
            return

        elapsed = (time.time() - self.last_audio_active_time) * 1000
        if elapsed >= 500: # Shortened grace period to match quick response
            print(f"\n[STATE] Speaking finished. Reverting to LISTENING")
            self.set_state(CallState.LISTENING)

    def shutdown(self):
        try:
            self.synthesizer.close()
        except:
            pass
