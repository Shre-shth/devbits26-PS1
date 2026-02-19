import os
import sys
import numpy as np
import scipy.signal
from enum import Enum

class CallState(Enum):
    LISTENING = 1
    THINKING = 2
    SPEAKING = 3

class AudioPreprocessor:
    def __init__(self, rate=16000, gain=3.0):
        self.rate = rate
        self.gain = gain
        # Bandpass: 80Hz - 7000Hz (Human voice range)
        self.sos = scipy.signal.butter(
            4, [80, 7000], btype='bandpass', fs=rate, output='sos'
        )
        self.zi = scipy.signal.sosfilt_zi(self.sos)

    def process(self, audio_chunk):
        audio_float = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        filtered, self.zi = scipy.signal.sosfilt(
            self.sos, audio_float, zi=self.zi
        )

        amplified = filtered * self.gain

        return np.clip(amplified, -32768, 32767).astype(np.int16)

def init_nvidia():
    """Initializes NVIDIA environment variables for cuBLAS/cuDNN."""
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib

        new_paths = [
            nvidia.cublas.lib.__path__[0],
            nvidia.cudnn.lib.__path__[0]
        ]

        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        needs_restart = False
        for p in new_paths:
            if p not in current_ld_path:
                current_ld_path = f"{p}:{current_ld_path}" if current_ld_path else p
                needs_restart = True

        if needs_restart:
            print(f"[INFO] Initializing environment with NVIDIA libraries: {new_paths}")
            os.environ["LD_LIBRARY_PATH"] = current_ld_path
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except ImportError:
        pass
