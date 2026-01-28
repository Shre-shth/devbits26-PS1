from piper import PiperVoice
import pyaudio
import queue
import threading
import os

class Synthesizer:
    def __init__(self, model_path="en_US-ryan-medium.onnx"):
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Piper model not found at {model_path}")
             
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
        
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()

    def _playback_worker(self):
        while True:
            audio_chunk = self.audio_queue.get()
            if audio_chunk is None:
                self.audio_queue.task_done()
                break
            
            try:
                self.is_playing = True
                self.stream.write(audio_chunk)
            except Exception as e:
                print(f"[TTS] Playback Error: {e}")
            finally:
                self.audio_queue.task_done()
                
            # If queue is empty, we are done playing
            # But wait, we loop immediately.
            # We can toggle is_playing here optionally, 
            # but is_active property covers queue check.
            pass

    def stop(self):
        """Clears the queue to stop playback immediately (Barge-In)."""
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

    def speak_stream(self, text_iterator):
        """
        Consumes text iterator, synthesizes, and plays.
        """
        self.is_playing = True
        buffer = ""
        punctuation = {'.', '?', '!', '\n'}
        
        for text_chunk in text_iterator:
            buffer += text_chunk
            if any(p in buffer for p in punctuation):
                # Simple sentence split
                to_speak = buffer
                buffer = ""
                self._synthesize_enqueue(to_speak)
        
        if buffer.strip():
            self._synthesize_enqueue(buffer)

    def synthesize_text(self, text):
        """
        Synthesizes a single text chunk and adds to audio queue.
        Useful for push-based streaming (Gemini -> Buffer -> Piper).
        """
        self.is_playing = True
        self._synthesize_enqueue(text)

    def _synthesize_enqueue(self, text):
        try:
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

    @property
    def is_active(self):
        """
        Returns True if audio is playing or queued.
        This includes the time the queue is non-empty AND the time we are actually writing bytes to the stream.
        """
        # We need a robust way to know if we are writing.
        # Check: Queue not empty OR (Queue empty but we haven't finished the current write?)
        # For simplicity, let's rely on queue size.
        # But if the worker pops the last item, queue is empty, but it's still playing for X ms.
        # Ideally we use a lock or flag.
        # Let's use self.audio_queue.unfinished_tasks?
        return self.audio_queue.unfinished_tasks > 0

    def wait_until_done(self):
        self.audio_queue.join()
        self.is_playing = False

    def close(self):
        self.audio_queue.put(None)
        self.playback_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
