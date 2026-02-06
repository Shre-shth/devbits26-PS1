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
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            # Optimize safety settings for speed
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # Initialize chat history (manual list management)
        # We process history manually to avoid "response not iterated" errors on interruption
        # Optimized system prompt: Concise, no formatting.
        self.history = [
            {"role": "user", "parts": ["[IMPORTANT: You are in a voice call. Do not use lists. Do not use bolding and do not use asterisks or other special characters. Keep answers under 15 words unless asked to elaborate.]"]},
            {"role": "model", "parts": ["Understood. I will be concise."]}
        ]

    def generate_response_stream(self, text):
        """
        Yields text chunks from Gemini using native SDK.
        """
        full_response = ""
        try:
            print(f"[Brain] Requesting stream for: '{text}'")
            
            # Add user message to history buffer
            self.history.append({"role": "user", "parts": [text]})
            
            # Generate content using stateless request
            response_stream = self.model.generate_content(
                self.history,
                stream=True
            )
            
            for chunk in response_stream:
                if chunk.text:
                    print(f"[Brain] Chunk: {chunk.text}", flush=True)
                    full_response += chunk.text
                    yield chunk.text
            
            # If completed successfully, add model response to history
            if full_response.strip():
                self.history.append({"role": "model", "parts": [full_response]})
                    
        except Exception as e:
            print(f"Brain Error: {e}", flush=True)
            # Rollback history on error (remove user message)
            if self.history and self.history[-1]["role"] == "user":
                 self.history.pop()
                 
            yield "Fuck off buddy !!"
