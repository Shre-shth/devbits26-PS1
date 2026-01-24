import os
import time
import sys
from langchain_google_genai import ChatGoogleGenerativeAI

# Force the model we want to test
MODEL_NAME = "gemini-flash-latest"

def test_connection():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY is not set.")
        return
    
    print(f"DEBUG: Using API Key ending in '...{api_key[-4:]}'")

    print(f"Testing connection to {MODEL_NAME}...")
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=0.7,
            google_api_key=api_key
        )
        
        start_time = time.time()
        print("Sending request: 'Hello, how are you?'")
        
        # Test streaming specifically since that's what we use
        chunk_count = 0
        first_token_time = None
        
        try:
            for chunk in llm.stream("Hello, how are you?"):
                current_time = time.time()
                if chunk_count == 0:
                    first_token_time = current_time
                    latency = first_token_time - start_time
                    print(f"\n[SUCCESS] First token received in {latency:.2f} seconds!")
                
                print(chunk.content, end="", flush=True)
                chunk_count += 1
            
            print(f"\n\nTotal chunks: {chunk_count}")
            print(f"Total time: {time.time() - start_time:.2f} seconds")
            
        except Exception as stream_err:
             print(f"\n[STREAM ERROR]: {stream_err}")
             # Fallback to invoke
             print("Retrying with non-streaming invoke...")
             response = llm.invoke("Hello")
             print(f"Invoke response: {response.content}")

    except Exception as e:
        print(f"\n[FATAL ERROR]: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_connection()
