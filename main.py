import os
import time
import queue
import pyaudio
import numpy as np
import threading
import re
from enum import Enum
from dotenv import load_dotenv

# Import modules
from vad import VADIterator
from transcriber import Transcriber
from brain import Brain
from synthesizer import Synthesizer

# Constants
RATE = 16000
CHUNK_SIZE = 512 # 32ms
FORMAT = pyaudio.paInt16

load_dotenv(override=True)
API_KEY = os.getenv("GOOGLE_API_KEY")

class CallState(Enum):
    LISTENING = 1
    THINKING = 2
    SPEAKING = 3

class VoiceBot:
    def __init__(self):
        print("Initializing modules...")
        self.vad = VADIterator(threshold=0.5, min_silence_duration_ms=500)
        self.transcriber = Transcriber()
        self.brain = Brain(API_KEY)
        self.synthesizer = Synthesizer()
        
        self.state = CallState.LISTENING
        self.buffer = bytearray()
        self.last_state_change = time.time()
        
        # Duration Check (Barge-In)
        self.barge_in_chunks = 0
        self.BARGE_IN_CHUNKS_THRESHOLD = 6 # ~200ms
        
        # Audio I/O
        self.p = pyaudio.PyAudio()
        self.recorder = self.p.open(
            format=FORMAT,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        # Notes: Gating thresholds are now internal to VAD or managed here for barge-in explicitly.
        self.BARGE_IN_ENERGY_THRESHOLD = 15000 
        self.STATE_TRANSITION_GRACE_MS = 1000

    def run(self):
        print("\n--- Low Latency Voice Bot Started ---")
        print(f"State: {self.state.name}")
        
        try:
            while True:
                # 1. Read Audio
                try:
                    audio_chunk = self.recorder.read(CHUNK_SIZE, exception_on_overflow=False)
                except IOError as e:
                    print(f"Audio Error: {e}")
                    continue
                
                # 2. Process Audio (State Machine)
                self.process_audio(audio_chunk)
                
                # 3. Check Playback (Auto-revert to LISTENING)
                if self.state == CallState.SPEAKING:
                     self.check_playback_status()
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.shutdown()

    def process_audio(self, audio_chunk):
        # Convert to numpy for VAD
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # --- ECHO CANCELLATION / GATING ---
        # If SPEAKING, ignore low energy audio (echo)
        if self.state == CallState.SPEAKING:
            # Grace period check
            if (time.time() - self.last_state_change) * 1000 < self.STATE_TRANSITION_GRACE_MS:
                return

            energy = np.abs(audio_int16).mean()
            # Threshold to ignore echo/noise
            if energy < self.BARGE_IN_ENERGY_THRESHOLD:
                # Ignore this chunk for barge-in
                return

        # --- VAD LOGIC ---
        # vad() returns dict or None (or dict with prob key)
        vad_event = self.vad(audio_int16)
        prob = vad_event.get('prob', 0.0)

        if self.state == CallState.LISTENING:
            self.handle_listening(audio_chunk, vad_event)
            
        elif self.state == CallState.SPEAKING or self.state == CallState.THINKING:
            self.handle_barge_in(audio_chunk, prob)

    def handle_listening(self, audio_chunk, vad_event):
        # The VADIterator handles the state.
        # We just need to react to events.
        
        # 1. Check EVENT: Start
        if vad_event.get('start') is not None:
             print("\n[Speech Detected]")
             # We could technically trim the buffer based on 'start' offset, 
             # but strictly extending is fine for minimal latency.
             
        # 2. Check EVENT: End
        if vad_event.get('end') is not None:
             print(f"\n[End of Turn] Buffer: {len(self.buffer)} bytes")
             
             self.set_state(CallState.THINKING)
             full_audio = bytes(self.buffer)
             self.buffer = bytearray()
             self.vad.reset_states() # Reset for next turn
             
             self.process_turn(full_audio)
             return

        # 3. If triggered (inside speech segment), BUFFER audio
        if self.vad.triggered:
             self.buffer.extend(audio_chunk)

    def handle_barge_in(self, audio_chunk, prob):
        # Interruption trigger
        if prob > self.vad.threshold:
            self.barge_in_chunks += 1
            
            # Check duration (6 chunks ~ 200ms)
            if self.barge_in_chunks >= self.BARGE_IN_CHUNKS_THRESHOLD:
                print("\n[!] BARGE-IN DETECTED - (Duration > 200ms)")
                
                # 1. Halt TTS
                self.synthesizer.stop()
                
                # 2. Transition -> LISTENING
                self.set_state(CallState.LISTENING)
                
                # 3. Capture this chunk!
                # Wait, we should probably have been capturing the *previous* chunks too if we want full fidelity.
                # But for now, starting from the trigger point + current buffer is okay.
                # Ideally, we should have a rolling buffer for VAD? 
                # For simplicity, extend buffer.
                self.buffer.extend(audio_chunk)
                
                # Note: Reset duration counter done in set_state(LISTENING)
        else:
             # Reset if continuous speech breaks
             self.barge_in_chunks = 0
            
    def set_state(self, new_state):
        self.state = new_state
        self.last_state_change = time.time()
        
        if new_state == CallState.LISTENING:
            self.vad.reset_states()
            # Also ensure buffer is clear to be safe
            self.buffer = bytearray()
            self.barge_in_chunks = 0 # Reset counter

    def process_turn(self, audio_data):
        # 1. Transcribe
        text = self.transcriber.transcribe(audio_data)
        print(f"User: {text}")
        
        if not text.strip():
            print("No speech recognized.")
            self.set_state(CallState.LISTENING)
            return

        # 2. Stream Generation (Daisy-Chain)
        
        # Thread/Async? 
        # Since we are in a loop check, we shouldn't block.
        # But 'generate_response_stream' is a generator.
        # We need to consume it without blocking the main VAD loop too much?
        # ACTUALLY: The VAD loop MUST run to detect Barge-In.
        # So specific generation must happen in a separate thread/task.
        
        threading.Thread(target=self._generation_worker, args=(text,), daemon=True).start()


    def _generation_worker(self, text):
        """
        Consumes Gemini stream, buffers sentences, sends to Piper.
        Runs in background thread so VAD loop continues.
        """
        print(f"Bot generating response for: {text}")
        
        sentence_buffer = ""
        # Regex to split by punctuation, keeping the punctuation.
        # Check for . ? ! ; followed by space or end of string
        split_pattern = r'(?<=[.?!;])\s+'
        
        try:
            stream = self.brain.generate_response_stream(text)
            
            for chunk in stream:
                # Check cancellation (if barge-in happened)
                if self.state == CallState.LISTENING:
                     print("[Generation] Cancelled (State is LISTENING)")
                     return

                sentence_buffer += chunk
                
                # Try to split buffer into sentences
                parts = re.split(split_pattern, sentence_buffer)
                
                if len(parts) > 1:
                    # We have at least one complete sentence
                    # All parts except the last one are definitely complete sentences
                    # (The split consumes the separator, but we used lookbehind so punctuation is kept)
                    
                    for i in range(len(parts) - 1):
                        to_speak = parts[i].strip()
                        if to_speak:
                             # Transition to SPEAKING if needed
                             if self.state == CallState.THINKING:
                                 self.set_state(CallState.SPEAKING)
                             
                             if self.state == CallState.SPEAKING:
                                 self.synthesizer.synthesize_text(to_speak)
                    
                    # The last part is the new buffer
                    sentence_buffer = parts[-1]

            # Leftovers
            if sentence_buffer and sentence_buffer.strip():
                 if self.state == CallState.THINKING:
                      self.set_state(CallState.SPEAKING)
                 if self.state == CallState.SPEAKING:
                      self.synthesizer.synthesize_text(sentence_buffer.strip())

        except Exception as e:
            print(f"Generation Error: {e}")
            self.set_state(CallState.LISTENING)
            
        print("[Generation] Done")
        # Note: State stays SPEAKING until TTS finishes? 
        # Ideally, we monitor TTS queue. If empty -> LISTENING.
        # We need a periodic check in main loop.
        
    def check_playback_status(self):
        """Called by main loop if in SPEAKING state"""
        # If TTS indicates it's done (queue empty), revert to LISTENING
        
        is_active = self.synthesizer.is_active
        # print(f"\r[DEBUG] TTS Active: {is_active} Queue: {self.synthesizer.audio_queue.unfinished_tasks}", end="", flush=True)
        
        if not is_active:
             # Add small grace period?
             if (time.time() - self.last_state_change) * 1000 > 1000: # Wait at least 1s before auto-reverting if idle
                 print("\n[State] Reverting to LISTENING")
                 self.set_state(CallState.LISTENING)

    def shutdown(self):
        self.recorder.stop_stream()
        self.recorder.close()
        self.p.terminate()
        self.synthesizer.close()

if __name__ == "__main__":
    bot = VoiceBot()
    bot.run()
