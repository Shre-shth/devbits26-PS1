import os
import sys

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from voice_bot import LLMClient

def test_profile_creation():
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

    # Simulate conversation
    turns = [
        "Hi, I am Tester.",
        "I need help finding a 2BHK apartment in New York.",
        "That is all for now, goodbye."
    ]

    print("\n--- Starting Conversation ---")
    full_conversation_history = []
    
    for turn_input in turns:
        print(f"\nUser: {turn_input}")
        response_text = ""
        for chunk in client.generate_response(turn_input):
            response_text += chunk
            # processing would happen here
        print(f"AI: {response_text}")
        full_conversation_history.append(response_text)

    # Check for profile in the last response (or any response)
    last_response = full_conversation_history[-1]
    
    if "<PROFILE>" in last_response and "</PROFILE>" in last_response:
        print("\n\nSUCCESS: Profile tag detected in response.")
        
        start_tag = "<PROFILE>"
        end_tag = "</PROFILE>"
        start_idx = last_response.find(start_tag) + len(start_tag)
        end_idx = last_response.find(end_tag)
        
        profile_content = last_response[start_idx:end_idx].strip()
        print(f"Extracted Profile:\n{profile_content}")
        
        # Verify JSON structure (simple check)
        if "Tester" in profile_content and "New York" in profile_content:
             print("Profile content looks correct.")
        else:
             print("WARNING: Profile content might be missing details.")
             
        # Simulate saving
        with open("user_profiles_test.txt", "a") as f:
            f.write(profile_content + "\n---\n")
        print("Profile saved to user_profiles_test.txt")

    else:
        print("\nFAILURE: No <PROFILE> tag found in the final response.")
        print("Full Final Response:")
        print(last_response)

if __name__ == "__main__":
    test_profile_creation()
