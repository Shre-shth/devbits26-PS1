import onnxruntime
import numpy as np
import os

class VADIterator:
    def __init__(self, 
                 model_path="silero_vad.onnx", 
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 300, # Tuned for responsiveness
                 speech_pad_ms: int = 30):
        """
        Class for stream imitation (ONNX version).
        
        Args:
            model_path: Path to .onnx Silero VAD model
            threshold: Speech threshold. Probabilities above this are SPEECH.
            sampling_rate: Currently supports 8000 or 16000
            min_silence_duration_ms: Milliseconds of silence to wait before ending a segment.
            speech_pad_ms: Final speech chunks are padded by this amount.
        """
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"VAD model not found at {model_path}")
             
        self.session = onnxruntime.InferenceSession(model_path)
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        # Constants
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        
        # State
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        
    def reset_states(self):
        """Resets the internal LSTM states of the model."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, audio_chunk):
        """
        Processes a single audio chunk.
        
        Args:
            audio_chunk (bytes or np.ndarray): Audio chunk.
            
        Returns:
            dict or None: {'start': int} when speech begins, {'end': int} when speech ends.
        """
        # Convert bytes to numpy if needed
        if isinstance(audio_chunk, (bytes, bytearray)):
            audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
            x = audio_int16.astype(np.float32) / 32768.0
        elif isinstance(audio_chunk, np.ndarray):
             if audio_chunk.dtype == np.int16:
                 x = audio_chunk.astype(np.float32) / 32768.0
             else:
                 x = audio_chunk
        else:
            raise TypeError("Audio chunk must be bytes or numpy array")
            
        # Add batch dimension: (Batch, Samples) -> (1, N)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        window_size_samples = x.shape[1]
        self.current_sample += window_size_samples

        # Inference: Get speech probability
        # Silero V5 expects: input, state, sr
        inputs = {
            'input': x,
            'state': self._state,
            'sr': np.array([self.sampling_rate], dtype=np.int64),
        }
        
        out, state = self.session.run(None, inputs)
        self._state = state
        speech_prob = out[0][0]

        # LOGIC: Detecting Speech Start
        # Note: Official spec uses "temp_end" to track silence gaps
        
        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0  # User resumed speaking during the silence wait window

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            # Return timestamp relative to current session (in samples)
            return {'start': self.current_sample - window_size_samples, 'prob': speech_prob}

        # LOGIC: Detecting Speech End
        if (speech_prob < (self.threshold - 0.15)) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample

            # Check if silence has persisted longer than min_silence_duration_ms
            if self.current_sample - self.temp_end >= self.min_silence_samples:
                
                # Speech has officially ended
                endTime = self.temp_end + self.speech_pad_samples
                
                self.triggered = False
                self.temp_end = 0
                return {'end': endTime, 'prob': speech_prob}

        return {'prob': speech_prob} # Return prob for debugging/logging
