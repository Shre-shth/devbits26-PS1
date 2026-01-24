import os
import pyaudio
from piper import PiperVoice
import wave

MODEL_PATH = "en_US-lessac-medium.onnx"

def test_tts():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    print("Loading Piper model...")
    voice = PiperVoice.load(MODEL_PATH)
    print("Piper model loaded.")
    print(f"Voice object type: {type(voice)}")
    print(f"Attributes: {dir(voice)}")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=22050,
        output=True
    )
    print("Audio stream opened.")

    text = "Hello, this is a test of the text to speech system."
    print(f"Synthesizing: '{text}'")

    try:
        # PiperVoice.synthesize returns an iterable of AudioChunk
        # each AudioChunk has audio_int16_bytes property
        for audio_chunk in voice.synthesize(text):
            stream.write(audio_chunk.audio_int16_bytes)
            
        print("Playback complete.")
        
    except Exception as e:
        print(f"Error during synthesis/playback: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    test_tts()
