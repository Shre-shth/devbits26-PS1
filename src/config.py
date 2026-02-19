import os
from dotenv import load_dotenv

load_dotenv(override=True)

RATE = 16000
CHUNK_SIZE = 320
API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAD_MODEL_PATH = os.path.join(BASE_DIR, "silero_vad.onnx")
DEFAULT_VOICE_MODEL = os.path.join(BASE_DIR, "en_US-amy-low.onnx")

VOICE_MODEL = os.getenv("VOICE_MODEL", DEFAULT_VOICE_MODEL)
if not os.path.isabs(VOICE_MODEL):
    VOICE_MODEL = os.path.join(BASE_DIR, VOICE_MODEL)
