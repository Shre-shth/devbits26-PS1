from faster_whisper import WhisperModel
import numpy as np

class Transcriber:
    def __init__(self, model_size="small.en"):
        device = "cpu"
        compute_type = "int8"
        
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                print("GPU Detected! Switching to CUDA (float16).")
                device = "cuda"
                compute_type = "float16"
        except Exception as e:
            print(f"GPU Check Failed: {e}. Defaulting to CPU.")

        print(f"Loading Whisper model: {model_size} on {device} ({compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Whisper model loaded.")

    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribes raw 16kHz PCM audio bytes.
        """
        # faster-whisper expects a NumPy array of float32, normalized to [-1, 1]
        # audio_data is raw int16 bytes
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        segments, info = self.model.transcribe(audio_np, beam_size=5) # Beam size 5 for accuracy (as requested)
        text = " ".join([segment.text for segment in segments]).strip()
        return text
