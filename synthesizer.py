from piper import PiperVoice
import pyaudio
import queue
import threading
import os
import numpy as np

class Synthesizer:
    def __init__(self, model_path="en_US-ryan-medium.onnx"):
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Piper model not found at {model_path}")
             
        self.voice = PiperVoice.load(model_path)
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.current_energy = 0.0 # Exposed for Echo Gate
        
        # Output stream
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=22050, # Piper default
            output=True
        )
        
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        
        # Interruption Control
        self.abort_event = threading.Event()

    def _playback_worker(self):
        CHUNK_SIZE = 1024 # Small chunk for responsive interruption
        
        while True:
            audio_data = self.audio_queue.get()
            if audio_data is None:
                self.audio_queue.task_done()
                break
            
            try:
                # We split the large audio_data into small chunks to allow frequent abort checks
                # play in chunks
                for i in range(0, len(audio_data), CHUNK_SIZE):
                    if self.abort_event.is_set():
                        break
                        
                    chunk = audio_data[i:i+CHUNK_SIZE]
                    
                    # Calculate Energy (RMS) on the fly for the gate
                    audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                    if len(audio_np) > 0:
                         self.current_energy = np.sqrt(np.mean(audio_np**2))
                    
                    self.is_playing = True
                    self.stream.write(chunk)
                    
            except Exception as e:
                print(f"[TTS] Playback Error: {e}")
            finally:
                self.current_energy = 0.0 # Reset
                self.audio_queue.task_done()
                
            pass

    def stop(self):
        """Clears the queue to stop playback immediately (Barge-In)."""
        self.abort_event.set() # Stop synthesis AND playback loop
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
            
    def reset(self):
        """Call this before starting a new turn"""
        self.abort_event.clear()

    def speak_stream(self, text_iterator):
        """
        Consumes text iterator, synthesizes, and plays.
        """
        self.is_playing = True
        buffer = ""
        punctuation = {'.', '?', '!', '\n'}
        
        for text_chunk in text_iterator:
            if self.abort_event.is_set(): return
            buffer += text_chunk
            if any(p in buffer for p in punctuation):
                # Simple sentence split
                to_speak = buffer
                buffer = ""
                self._synthesize_enqueue(to_speak)
        
        if buffer.strip() and not self.abort_event.is_set():
            self._synthesize_enqueue(buffer)

    def synthesize_text(self, text):
        """
        Synthesizes a single text chunk and adds to audio queue.
        Useful for push-based streaming (Gemini -> Buffer -> Piper).
        """
        self.is_playing = True
        self._synthesize_enqueue(text)

    def _synthesize_enqueue(self, text):
        if self.abort_event.is_set():
             return

        try:
            self.is_synthesizing = True # Start synthesis
            print(f"[TTS] Synthesizing: '{text}'")
            # Piper stream
            # The synthesize method returns a generator of audio chunks (raw PCM by default?)
            # We need to pass a speaker_id if multi-speaker, or 0/None.
            # From usage: synthesize(text, speaker_id=0, length_scale=1.0, ...)
            # Let's try passing just text and an empty speaker_id which seems to be required by older bindings or simple 0.
            
            # Note: The test script used list() as second arg? 
            # Let's try standard usage: stream = self.voice.synthesize(text)
            
            stream = self.voice.synthesize(text)
            
            for audio_chunk in stream:
                 if self.abort_event.is_set():
                     print("[TTS] Aborted during synthesis loop")
                     return

                 # Extract bytes from AudioChunk
                 # It seems to be a property or method based on dir() output
                 # audio_int16_bytes is likely what we want. 
                 # Let's try to access it as a property first based on commonly used dataclasses, 
                 # but since dir showed it, it exists.
                 data = audio_chunk.audio_int16_bytes
                 self.audio_queue.put(data)
            print("[TTS] Audio enqueued")
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.is_synthesizing = False # End synthesis

    @property
    def is_active(self):
        """
        Returns True if audio is playing, queued, OR being synthesized.
        """
        # We need a robust way to know if we are writing.
        # Check: Queue not empty OR (Queue empty but we haven't finished the current write?)
        # For simplicity, let's rely on queue size.
        # But if the worker pops the last item, queue is empty, but it's still playing for X ms.
        # Ideally we use a lock or flag.
        # Let's use self.audio_queue.unfinished_tasks?
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
