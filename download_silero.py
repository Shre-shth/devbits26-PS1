import os
import urllib.request

MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
OUTPUT_FILE = "silero_vad.onnx"

def download_model():
    print(f"Downloading Silero VAD model from {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, OUTPUT_FILE)
        print(f"Successfully downloaded to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Failed to download model: {e}")

if __name__ == "__main__":
    download_model()
