import sys
try:
    from piper import PiperVoice
    print("PiperVoice found.")
    print("Dir:", dir(PiperVoice))
    print("Synthesize doc:", PiperVoice.synthesize.__doc__)
except ImportError:
    print("PiperVoice not found")
except Exception as e:
    print(e)
