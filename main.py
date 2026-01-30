import os
import time
import queue
import pyaudio
import numpy as np
import threading
import re
from enum import Enum
from dotenv import load_dotenv
import scipy.signal

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

class AudioPreprocessor:
    """
    Implements efficient Spectral Filtering (Bandpass) using Scipy.
    This approximates the 'Spectral Subtraction' need by removing 
    out-of-band noise (rumble/hiss) that inflates energy calculations.
    """
    def __init__(self, rate=16000):
        self.rate = rate
        # Bandpass: 80Hz - 7000Hz (Human voice range)
        # Using SOS (Second Order Sections) for stability
        self.sos = scipy.signal.butter(4, [80, 7000], btype='bandpass', fs=rate, output='sos')
        self.zi = scipy.signal.sosfilt_zi(self.sos)

    def process(self, audio_chunk):
        """
        Filters the audio chunk in real-time.
        Returns filtered numpy array (int16).
        """
        # Convert to float for DSP
        audio_float = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        
        # Apply Filter
        filtered, self.zi = scipy.signal.sosfilt(self.sos, audio_float, zi=self.zi)
        
        # Convert back to int16
        return np.clip(filtered, -32768, 32767).astype(np.int16)

class VoiceBot:
    def __init__(self):
        print("Initializing modules...")
        self.vad = VADIterator(threshold=0.5, min_silence_duration_ms=500)
        self.transcriber = Transcriber()
        self.brain = Brain(API_KEY)
        self.synthesizer = Synthesizer()
        self.audio_processor = AudioPreprocessor(RATE)
        
        self.state = CallState.LISTENING
        self.buffer = bytearray()
        self.last_state_change = time.time()
        
        # Duration Check (Barge-In)
        self.barge_in_chunks = 0
        self.BARGE_IN_CHUNKS_THRESHOLD = 8 # ~250ms (Faster interruption)
        self.pre_barge_buffer = bytearray() # Buffer to hold chunks while confirming barge-in
        
        # Audio I/O
        self.p = pyaudio.PyAudio()
        self.recorder = self.p.open(
            format=FORMAT,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        # --- Logic 7.2: Server-Side Gating ---
        # "If CallState == SPEAKING, ignore... energy < threshold"
        # Since we applied spectral filtering, we can maintain a reasonable threshold.
        # Original: 15000 (Very high). 
        # New: 10000 (Still high to block echo, but filtered audio is cleaner)
        self.ECHO_GATE_THRESHOLD = 10000 
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
        # --- 1. Spectral Filtering (Improve SNR) ---
        # Logic 7.2: "Spectral Subtraction" (Approximated via Bandpass)
        audio_int16 = self.audio_processor.process(audio_chunk)
        
        # --- 2. Echo Gate (Logic 7.2) ---
        if self.state == CallState.SPEAKING:
            # Grace period check
            if (time.time() - self.last_state_change) * 1000 < self.STATE_TRANSITION_GRACE_MS:
                return

            energy = np.abs(audio_int16).mean()
            
            # Logic: "Only trigger barge-in if input volume is significantly louder than echo"
            output_energy = self.synthesizer.current_energy
            dynamic_threshold = max(self.ECHO_GATE_THRESHOLD, output_energy * 0.8) # 0.8 factor (Balanced)
            
            if energy < dynamic_threshold:
                # Still keep this commented to avoid spamming 100 lines/sec, but maybe print every N frames or if energy > 1000
                if energy > 2000:
                    # Debug log only if there is *some* sound, to see why it's rejected
                    print(f"\r[Echo Gate] Ignored. Energy: {energy:.1f} < {dynamic_threshold:.1f} (Out: {output_energy:.1f})      ", end="", flush=True)
                return
            else:
                print(f"\r[Echo Gate] OPEN! Energy: {energy:.1f} > {dynamic_threshold:.1f} (Out: {output_energy:.1f})      ", end="", flush=True)

        # --- 3. VAD Logic ---
        # vad() returns dict or None (or dict with prob key)
        # Note: We pass the FILTERED audio to VAD for better accuracy
        vad_event = self.vad(audio_int16)
        prob = vad_event.get('prob', 0.0)

        if self.state == CallState.LISTENING:
            self.handle_listening(audio_chunk, vad_event)
            
        elif self.state == CallState.SPEAKING or self.state == CallState.THINKING:
            self.handle_barge_in(audio_chunk, prob)

    def handle_listening(self, audio_chunk, vad_event):
        # 1. Check EVENT: Start
        if vad_event.get('start') is not None:
             print("\n[Speech Detected]")
             
        # 2. Check EVENT: End
        if vad_event.get('end') is not None:
             print(f"\n[End of Turn] Buffer: {len(self.buffer)} bytes")
             
             self.set_state(CallState.THINKING)
             full_audio = bytes(self.buffer)
             self.buffer = bytearray()
             self.vad.reset_states() # Reset for next turn
             
             self.process_turn(full_audio)
             return

        if self.vad.triggered:
             self.buffer.extend(audio_chunk)

    def handle_barge_in(self, audio_chunk, prob):
        # Interruption trigger
        if prob > self.vad.threshold:
            self.barge_in_chunks += 1
            self.pre_barge_buffer.extend(audio_chunk) # Capture candidate audio
            
            # Check duration
            if self.barge_in_chunks >= self.BARGE_IN_CHUNKS_THRESHOLD:
                print("\n[!] BARGE-IN DETECTED - (Duration > 480ms)")
                
                # 1. Halt TTS
                try:
                    self.synthesizer.stop()
                except Exception as e:
                    print(f"Barge-In Stop Error: {e}")
                
                # 2. Transition -> LISTENING
                self.set_state(CallState.LISTENING)
                
                # 3. Capture this chunk!
                # IMPORTANT: Flush the pre_barge_buffer which contains the "lost" start words
                self.buffer.extend(self.pre_barge_buffer)
                self.pre_barge_buffer = bytearray() # Clear
                # self.buffer.extend(audio_chunk) # Covered by pre_buffer
                
                # Force VAD trigger state so handle_listening continues recording
                self.vad.triggered = True 
        else:
             # Reset if continuous speech breaks
             self.barge_in_chunks = 0
             self.pre_barge_buffer = bytearray()
            
    def set_state(self, new_state):
        self.state = new_state
        self.last_state_change = time.time()
        
        if new_state == CallState.LISTENING:
            self.vad.reset_states()
            # Also ensure buffer is clear to be safe
            self.buffer = bytearray()
            self.pre_barge_buffer = bytearray()
            self.barge_in_chunks = 0 # Reset counter

    def process_turn(self, audio_data):
        # 0. Reset Synthesizer (Clear abort flag from potential previous interruption)
        self.synthesizer.reset()

        # 1. Transcribe
        text = self.transcriber.transcribe(audio_data)
        print(f"User: {text}")
        
        if not text.strip():
            print("No speech recognized.")
            self.set_state(CallState.LISTENING)
            return

        # 2. Stream Generation
        threading.Thread(target=self._generation_worker, args=(text,), daemon=True).start()

    def _generation_worker(self, text):
        """
        Consumes Gemini stream, buffers sentences, sends to Piper.
        """
        print(f"Bot generating response for: {text}")
        
        sentence_buffer = ""
        # Regex to split by punctuation
        split_pattern = r'(?<=[.?!;])\s+'
        
        try:
            stream = self.brain.generate_response_stream(text)
            
            for chunk in stream:
                if self.state == CallState.LISTENING:
                     print("[Generation] Cancelled (State is LISTENING)")
                     return

                sentence_buffer += chunk
                parts = re.split(split_pattern, sentence_buffer)
                
                if len(parts) > 1:
                    for i in range(len(parts) - 1):
                        to_speak = parts[i].strip()
                        if to_speak:
                             if self.state == CallState.THINKING:
                                 self.set_state(CallState.SPEAKING)
                             
                             if self.state == CallState.SPEAKING:
                                 self.synthesizer.synthesize_text(to_speak)
                    
                    sentence_buffer = parts[-1]

            if sentence_buffer and sentence_buffer.strip():
                 if self.state == CallState.THINKING:
                      self.set_state(CallState.SPEAKING)
                 if self.state == CallState.SPEAKING:
                      self.synthesizer.synthesize_text(sentence_buffer.strip())

        except Exception as e:
            print(f"Generation Error: {e}")
            self.set_state(CallState.LISTENING)
            
        print("[Generation] Done")
        
    def check_playback_status(self):
        """Called by main loop if in SPEAKING state"""
        is_active = self.synthesizer.is_active
        
        if not is_active:
             # Immediate revert
             print("\n[State] Reverting to LISTENING")
             self.set_state(CallState.LISTENING)

    def shutdown(self):
        try:
            if self.recorder.is_active():
                self.recorder.stop_stream()
            self.recorder.close()
            self.p.terminate()
            self.synthesizer.close()
        except:
            pass

if __name__ == "__main__":
    bot = VoiceBot()
    bot.run()
