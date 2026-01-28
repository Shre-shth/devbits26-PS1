import os
import time
import queue
import threading
import numpy as np
import pyaudio
import onnxruntime
from faster_whisper import WhisperModel
import wave
import collections

# Audio Configuration
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024 # frames per buffer

class AudioRecorder:
    def __init__(self, input_device_index=None, silence_threshold_ms=600, speech_pad_ms=500, energy_threshold=1000):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        
        self.silence_threshold_ms = silence_threshold_ms
        self.speech_pad_ms = speech_pad_ms
        self.energy_threshold = energy_threshold
        self.q = queue.Queue()
        self.running = False
        self.listening = False
        self.paused = False

    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False

    def listen(self):
        print(f"Listening (Energy Threshold: {self.energy_threshold})...")
        self.running = True
        self.listening = True
        
        # Duration represented by CHUNK_SIZE
        chunk_duration_ms = (CHUNK_SIZE / RATE) * 1000
        
        num_silent_chunks = 0
        max_silent_chunks = int(self.silence_threshold_ms / chunk_duration_ms)
        speech_started = False
        frames = []
        
        # Pre-speech buffer to catch the start of words
        pre_speech_buffer = collections.deque(maxlen=10)
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                try:
                    # Flush stream buffer to avoid old audio
                    while self.stream.get_read_available() > CHUNK_SIZE:
                        self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                except Exception:
                    pass
                continue
                
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                # Calculate energy
                audio_np = np.frombuffer(data, dtype=np.int16)
                energy = np.abs(audio_np).mean()

                # Simple VAD based on Energy
                is_speech = energy > self.energy_threshold

                # VU Meter
                status = "SPEECH" if is_speech else "......"
                print(f"\rVol: {energy:6.1f} [{status}]", end="", flush=True)

                if is_speech:
                    if not speech_started:
                        print(f"\rVol: {energy:6.1f} [START]   ")
                        # Prepend buffer
                        frames.extend(pre_speech_buffer)
                        pre_speech_buffer.clear()
                        
                    speech_started = True
                    num_silent_chunks = 0
                    frames.append(data)
                elif speech_started:
                    num_silent_chunks += 1
                    frames.append(data)
                    
                    if num_silent_chunks > max_silent_chunks:
                        print(f"\rVol: {energy:6.1f} [END]     ")
                        
                        # Filter short clicks (e.g. < 200ms)
                        if len(frames) > 6:
                            # Return audio data
                            audio_data = b''.join(frames[:-max_silent_chunks]) 
                            self.q.put(audio_data)
                        else:
                            print("Ignored short noise.")
                        
                        # Reset for next phrase
                        speech_started = False
                        frames = []
                        num_silent_chunks = 0
                else:
                    # Buffer silence for pre-roll
                    pre_speech_buffer.append(data)
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
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.messages import HumanMessage, AIMessage

        if not api_key:
            raise ValueError("Google API Key is required for Gemini.")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            temperature=0.7, 
            google_api_key=api_key
        )
        
        # Store conversation history
        self.chat_history = []
        
        # System prompt for General Assistant
        system_template = """You should act as a customer care agent for general purpose. ask specific questions to the user regarding the query.
Constraints & Quality Control
Low Latency: Keep your responses concise (under 2 sentences) to minimize processing time and avoid awkward silences.
No Hallucinations: If you do not have specific data on a project’s price, offer to have a human representative send the latest brochure instead of making up numbers.
Interruption Handling: If the user starts speaking while you are responding, acknowledge the interruption immediately and pivot to their new question.
also dont generate response with specific symbols like *,#,etc.
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_response(self, text):
        from langchain_core.messages import HumanMessage, AIMessage
        
        full_response = ""
        for chunk in self.chain.stream({"input": text, "chat_history": self.chat_history}):
            full_response += chunk
            yield chunk
            
        # Update history after full response is generated
        # We store the full response in history, even with the profile tag, so the bot "remembers" it outputted it.
        self.chat_history.append(HumanMessage(content=text))
        self.chat_history.append(AIMessage(content=full_response))

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
            for audio_chunk in self.voice.synthesize(text):
                self.audio_queue.put(audio_chunk.audio_int16_bytes)
        except Exception as e:
            print(f"TTS Error: {e}")

    def _unused_method(self):
        # Placeholder to maintain structure if needed, but we removed the profile filtering
        pass

    def close(self):
        self.audio_queue.put(None) # Signal thread to stop
        self.playback_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class VoiceBot:
    def __init__(self, device_index=None):
        from dotenv import load_dotenv
        # Force override to ensure we use the key from .env, ignoring stale shell variables
        load_dotenv(override=True)
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
             print("Please set GOOGLE_API_KEY environment variable or in .env file.")
             sys.exit(1)
             
        print(f"DEBUG: VoiceBot using API Key ending in '...{api_key[-4:]}'")
        self.recorder = AudioRecorder(input_device_index=device_index)
        self.transcriber = Transcriber(model_size="small.en", device="cpu", compute_type="int8") # Use 'medium' on GPU
        self.llm = LLMClient(api_key)
        self.synthesizer = Synthesizer()

        




    def start(self):
        play_thread = threading.Thread(target=self.recorder.listen)
        play_thread.start()
        
        print("\n\n--- Voice Bot Instance Started (Customer Care Mode) ---")
        print("Speak into your microphone...")
        
        # Buffer to accumulate audio across "thinking pauses"
        self.turn_audio_buffer = bytearray()

        
        
        try:
            while True:
                # 1. Capture (Wait for silence)
                chunk_audio = self.recorder.get_audio()
                
                # Append to current turn buffer
                self.turn_audio_buffer.extend(chunk_audio)
                
                # Pause recording while processing
                self.recorder.pause()
                
                try:
                    # 2. Transcribe (Provisional)
                    print("Checking for completeness...", end=" ", flush=True)
                    # Convert bytearray to bytes for transcriber
                    full_turn_audio = bytes(self.turn_audio_buffer)
                    user_text, lang = self.transcriber.transcribe(full_turn_audio)
                    print(f"User ({lang}): {user_text}")
                    
                    if len(user_text.strip()) < 1:
                        # Probably just noise
                        print("Noise detected, ignoring...")
                        # We might want to clear buffer if it's just pure noise to avoid drift, 
                        # but for now let's keep accumulating just in case it was a quiet start.
                        self.recorder.resume()
                        continue
                        

                    
                    # --- Phrase is Complete -> Process Response ---
                    
                    # Clear buffer for next turn
                    self.turn_audio_buffer = bytearray()

                    # 4. LLM & TTS Pipeline
                    print("Gemini responding...", flush=True)
                    
                    try:
                        self.full_accumulated_response = ""
                        
                        def text_generator():
                            first_token_received = False
                            start_time = time.time()
                            
                            for chunk in self.llm.generate_response(user_text):
                                self.full_accumulated_response += chunk
                                
                                # Normal text
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
                        error_msg = str(e)
                        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                            print(f"\n[AI GEN LIMIT]: Quota exceeded. Please wait a moment.")
                            self.synthesizer.speak_text("I'm exhausted. Please give me a minute to recover.")
                        else:
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
