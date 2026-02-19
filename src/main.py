#!/usr/bin/env python3
import threading
from utils import init_nvidia
from ari_controller import ARIController
from voice_bot import VoiceBot

# Init NVIDIA libraries
init_nvidia()

if __name__ == "__main__":
    bot = VoiceBot()
    bot.ari = ARIController(bot)
    bot.ari.start()

    threading.Thread(target=bot.run, daemon=True).start()

    print("[MAIN] Bot is running. Type 'call' to trigger outbound call.")
    
    while True:
        try:
            cmd = input("Type 'call' to dial 6002: ")
            if cmd.strip() == "call":
                bot.ari.make_outbound_call("6002")
        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except EOFError:
            break