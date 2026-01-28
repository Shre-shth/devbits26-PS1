import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class Brain:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Google API Key is required for Gemini.")

        # Configure the native SDK
        genai.configure(api_key=api_key)
        
        # Use valid model name
        # Use valid model name
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize chat history (native format)
        # History is managed by the chat session object in this SDK,
        # but for simplicity we can manage it manually or use start_chat().
        # Let's use start_chat(history=[]) for automatic history management.
        self.chat_session = self.model.start_chat(history=[
            {"role": "user", "parts": ["You are a helpful customer service voice bot. Keep your responses concise (1-2 sentences). Do not use markdown symbols like * or #. If interrupted, stop immediately."]},
            {"role": "model", "parts": ["Understood. I will be concise and helpful."]}
        ])

    def generate_response_stream(self, text):
        """
        Yields text chunks from Gemini using native SDK.
        """
        full_response = ""
        try:
            print(f"[Brain] Requesting stream for: '{text}'")
            
            # Use the chat session to send a message
            # stream=True returns a generator immediately
            response_stream = self.chat_session.send_message(text, stream=True)
            
            for chunk in response_stream:
                if chunk.text:
                    print(f"[Brain] Chunk: {chunk.text}", flush=True)
                    full_response += chunk.text
                    yield chunk.text
                    
        except Exception as e:
            print(f"Brain Error: {e}", flush=True)
            yield " Fuck off"
            # In case of error, we might want to manually revert the history? 
            # The native SDk handles failed turns gracefully usually.
            
        # Note: History is automatically updated by the chat_session object!
