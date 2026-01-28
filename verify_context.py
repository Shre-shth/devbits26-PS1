import os
import sys

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from voice_bot import LLMClient

def test_context():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set.")
        return

    print("Initializing LLMClient...")
    try:
        client = LLMClient(api_key)
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        return

    # Turn 1
    print("\n--- Turn 1 ---")
    user_input_1 = "My name is AutoTester."
    print(f"User: {user_input_1}")
    response_1 = ""
    for chunk in client.generate_response(user_input_1):
        print(chunk, end="", flush=True)
        response_1 += chunk
    print("\n")

    # Turn 2
    print("--- Turn 2 ---")
    user_input_2 = "What is my name?"
    print(f"User: {user_input_2}")
    response_2 = ""
    for chunk in client.generate_response(user_input_2):
        print(chunk, end="", flush=True)
        response_2 += chunk
    print("\n")

    # Verification
    if "AutoTester" in response_2:
        print("SUCCESS: Context retained. Bot remembered the name.")
    else:
        print("FAILURE: Context NOT retained. Bot did not mention 'AutoTester'.")

if __name__ == "__main__":
    test_context()
