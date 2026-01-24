import os
import urllib.request

MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx?download=true"
CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json?download=true"

MODEL_FILE = "en_US-ryan-medium.onnx"
CONFIG_FILE = "en_US-ryan-medium.onnx.json"

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        download_file(MODEL_URL, MODEL_FILE)
    else:
        print(f"{MODEL_FILE} already exists.")

    if not os.path.exists(CONFIG_FILE):
        download_file(CONFIG_URL, CONFIG_FILE)
    else:
        print(f"{CONFIG_FILE} already exists.")
