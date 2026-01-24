import os
import time
import queue
import threading
import numpy as np
import pyaudio
import webrtcvad
from faster_whisper import WhisperModel
import wave
import collections

# Audio Configuration
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_DURATION_MS = 30  # Supports 10, 20, 30
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)

class AudioRecorder:
    def __init__(self, input_device_index=None, silence_threshold_ms=1000, speech_pad_ms=500, energy_threshold=2000):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        self.vad = webrtcvad.Vad(3)  # Aggressiveness: 0-3 (Increased to 3 to filter noise)
        self.silence_threshold_ms = silence_threshold_ms
        self.speech_pad_ms = speech_pad_ms
        self.energy_threshold = energy_threshold
        self.q = queue.Queue()
        self.running = False
        self.listening = False
        self.paused = False

    def is_speech(self, frame):
        try:
            return self.vad.is_speech(frame, RATE)
        except Exception as e:
            print(f"VAD Error: {e}")
            return False
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False

    def listen(self):
        print("Listening...")
        self.running = True
        self.listening = True
        
        num_silent_chunks = 0
        max_silent_chunks = int(self.silence_threshold_ms / CHUNK_DURATION_MS)
        speech_started = False
        frames = []

        while self.running:
            if self.paused:
                # Keep draining the buffer while paused so we don't have old audio when resuming
                try:
                    while self.stream.get_read_available() > CHUNK_SIZE:
                        self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                except Exception:
                    pass
                
                time.sleep(0.1)
                continue
                
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                # Calculate energy (amplitude)
                audio_np = np.frombuffer(data, dtype=np.int16)
                energy = np.abs(audio_np).mean()
                
                # Noise Gate: Force silence if energy is too low, regardless of VAD
                # Lowered threshold to 20 since user reports low volume detection
                is_speech = self.is_speech(data)
                
                if energy < self.energy_threshold:
                    is_speech = False

                # VU Meter Visualization
                status = "SPEECH" if is_speech else "......"
                bars = "#" * int(energy / 50) # Scale down for visualization
                print(f"\rVol: {energy:6.1f} [{status}] |{bars.ljust(50)}|", end="", flush=True)

                if is_speech:
                    if not speech_started:
                        print(f"\nSpeech detection started (Energy: {energy:.2f})") # Newline to break VU meter line
                    speech_started = True
                    num_silent_chunks = 0
                    frames.append(data)
                elif speech_started:
                    num_silent_chunks += 1
                    frames.append(data)
                    if num_silent_chunks > max_silent_chunks:
                        print("Silence detected, processing speech...")
                        # Return audio data
                        audio_data = b''.join(frames[:-max_silent_chunks]) # Trim end silence
                        self.q.put(audio_data)
                        
                        # Reset for next phrase
                        speech_started = False
                        frames = []
                        num_silent_chunks = 0
                else:
                    # Keep a small buffer of pre-speech audio? 
                    # For now just ignore silence before speech
                    pass
                
            except Exception as e:
                print(f"Error in listen loop: {e}")
                break

    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def get_audio(self):
        return self.q.get()

def list_input_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    print("\n--- Available Input Devices ---")
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("-------------------------------\n")
    p.terminate()

class Transcriber:
    def __init__(self, model_size="medium", device="cpu", compute_type="int8"):
        print(f"Loading Whisper model: {model_size} on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Whisper model loaded.")

    def transcribe(self, audio_data):
        # faster-whisper expects a NumPy array of float32, normalized to [-1, 1]
        # audio_data is raw int16 bytes
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        segments, info = self.model.transcribe(audio_np, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        return text, info.language

class LLMClient:
    def __init__(self, api_key):
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        if not api_key:
            raise ValueError("Google API Key is required for Gemini.")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest", 
            temperature=0.7, 
            google_api_key=api_key
        )
        
        # System prompt to keep responses concise for voice conversation
        system_template = """You are a helpful voice assistant. 
        Keep your responses concise and conversational, suitable for text-to-speech. 
        Avoid markdown formatting like bullets or bold text unless necessary for emphasis in speech.
        Limit your responses to 1-2 sentences unless asked for more detail."""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{input}"),
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_response(self, text):
        return self.chain.stream({"input": text})

class Synthesizer:
    def __init__(self, model_path="en_US-ryan-medium.onnx"):
        from piper import PiperVoice
        import wave
        
        self.model_path = model_path
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Piper model not found at {model_path}. Run download_models.py first.")
             
        self.voice = PiperVoice.load(model_path)
        self.audio_queue = queue.Queue()
        self.is_playing = False
        
        # Output stream
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=22050, # Piper default
            output=True
        )
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()

    def _playback_worker(self):
        while True:
            audio_chunk = self.audio_queue.get()
            if audio_chunk is None:
                break
            self.stream.write(audio_chunk)
            self.audio_queue.task_done()

    def speak_stream(self, text_iterator):
        """
        Consumes text iterator, buffers text into sentences, and synthesizes them.
        """
        self.is_playing = True
        buffer = ""
        # Simple punctuation detection for sentence splitting
        punctuation = {'.', '?', '!', '\n'}
        
        for text_chunk in text_iterator:
            buffer += text_chunk
            
            # Check if we have a sentence ending
            if any(p in buffer for p in punctuation):
                sentences = []
                current_sentence = ""
                for char in buffer:
                    current_sentence += char
                    if char in punctuation:
                        sentences.append(current_sentence.strip())
                        current_sentence = ""
                
                # The last part is incomplete, keep it in buffer
                if current_sentence:
                    buffer = current_sentence
                else:
                    buffer = "" # All perfectly split
                
                # Synthesize complete sentences
                for sentence in sentences:
                    if sentence:
                        self._synthesize_and_play(sentence)
            # Fail-safe: If buffer gets too long without punctuation, force a split
            elif len(buffer) > 80 and ' ' in buffer: 
                 # Find the last space to split reasonably safely
                 last_space = buffer.rfind(' ')
                 to_speak = buffer[:last_space]
                 buffer = buffer[last_space:]
                 self._synthesize_and_play(to_speak)
        
        # Play any remaining text in buffer
        if buffer.strip():
            self._synthesize_and_play(buffer.strip())
    
            self._synthesize_and_play(buffer.strip())
    
    def wait_until_done(self):
        """Blocks until all audio in the queue has been played."""
        self.audio_queue.join()
        self.is_playing = False
    
    def speak_text(self, text):
        """Helper to speak a simple string."""
        self._synthesize_and_play(text)

    def _synthesize_and_play(self, text):
        try:
            # Piper synthesize yields audio chunks
            # Piper synthesize yields audio chunks
            for audio_chunk in self.voice.synthesize(text):
                self.audio_queue.put(audio_chunk.audio_int16_bytes)
        except Exception as e:
            print(f"TTS Error: {e}")

    def close(self):
        self.audio_queue.put(None) # Signal thread to stop
        self.playback_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class VoiceBot:
    def __init__(self, device_index=None):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
             print("Please set GOOGLE_API_KEY environment variable.")
             sys.exit(1)
             
        print(f"DEBUG: VoiceBot using API Key ending in '...{api_key[-4:]}'")
        self.recorder = AudioRecorder(input_device_index=device_index)
        self.transcriber = Transcriber(model_size="small.en", device="cpu", compute_type="int8") # Use 'medium' on GPU
        self.llm = LLMClient(api_key)
        self.synthesizer = Synthesizer()
        
    def start(self):
        play_thread = threading.Thread(target=self.recorder.listen)
        play_thread.start()
        
        print("\n\n--- Voice Bot Instance Started ---")
        print("Speak into your microphone...")
        
        try:
            while True:
                # 1. Capture
                audio_data = self.recorder.get_audio()
                
                # Pause recording while processing/speaking
                self.recorder.pause()
                
                try:
                    # 2. Transcribe
                    print("Transcribing...", end=" ", flush=True)
                    user_text, lang = self.transcriber.transcribe(audio_data)
                    print(f"User ({lang}): {user_text}")
                    
                    if len(user_text.strip()) < 2:
                        continue
    
                    # 3. LLM & TTS Pipeline
                    print("Gemini responding...", flush=True)
                    
                    try:
                        def text_generator():
                            first_token_received = False
                            start_time = time.time()
                            for chunk in self.llm.generate_response(user_text):
                                if chunk:
                                    if not first_token_received:
                                        print(f"\n[Perf] Time to First Token: {time.time() - start_time:.2f}s")
                                        first_token_received = True
                                    print(chunk, end="", flush=True)
                                    yield chunk
                            print() # Newline
                        
                        self.synthesizer.speak_stream(text_generator())
                        self.synthesizer.wait_until_done()
                    except Exception as e:
                        print(f"\n[AI ERROR]: {e}")
                        self.synthesizer.speak_text("Sorry, I'm having trouble connecting to my brain right now.")
                    
                finally:
                    # Resume recording
                    print("\nListening...")
                    self.recorder.resume()
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.recorder.stop()
            self.synthesizer.close()
            play_thread.join()

if __name__ == "__main__":
    import sys
    
    # Check for device ID argument
    device_idx = None
    if len(sys.argv) > 1:
        try:
            device_idx = int(sys.argv[1])
        except ValueError:
            print("Usage: python voice_bot.py [device_index]")
            sys.exit(1)
            
    list_input_devices()
    
    if device_idx is None:
        print("To use a specific device, run: python voice_bot.py <device_index>")
        print("Using default device...")
        
    bot = VoiceBot(device_index=device_idx)
    bot.start()
