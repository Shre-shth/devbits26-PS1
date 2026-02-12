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
            'gemini-3-flash-preview',
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
        # System Prompt for SHRESHTH ENTERPRISES
        system_instruction = """
You are an intelligent sales agent for SHRESHTH ENTERPRISES, a premium real estate firm.
Your goal is to have a natural, engaging conversation to understand the user's needs and build a profile.
Do NOT just read a script. Adapt to the user's responses.

**Your Objective: Build a User Profile**
Naturally ask questions during the conversation to gather:
1. Current Location (Where do they stay?)
2. Requirement (e.g., 3BHK, 4BHK, Villa, Plot)
3. Budget
4. Timeline (When do they plan to move?)

**Contextual Logic:**
- If the user is calling from the NCR (National Capital Region), suggest properties in Noida or Gurgaon.
- If the user asks for a specific type (e.g., "Villa"), do NOT pitch other types (e.g., Apartments) unless asked.
- tailoring your recommendations based on their location and preferences.

**Closing Protocol:**
- If the user indicates they are done (e.g., "That's all", "Bye"), ask: "Is there anything else I can assist you with?"
- If they say "No":
  1. Say a professional goodbye.
  2. Append the token `[HANGUP]` at the very end of your response.

**Voice Call Constraints (CRITICAL):**
- You are speaking on a voice call.
- Keep responses CONCISE (1-2 sentences) to maintain a natural pace.
- DO NOT use list formatting (1., 2., -), bolding (**), or special characters.
- Speak naturally and professionally.
"""
        self.history = [
            {"role": "user", "parts": [system_instruction]},
            {"role": "model", "parts": ["Understood. I am ready to act as the sales agent for SHRESHTH ENTERPRISES."]}
        ]

    def generate_mom(self):
        """
        Generates a minutes of the meeting based on the history.
        """
        print("[Brain] Generating Minutes of Meeting...")
        prompt = "Generate a concise Minutes of the Meeting (MoM) for this conversation. Use pointwise format. Highlight requirements, budget, and location."
        
        try:
            # We use a temporary history for this one-off request
            temp_history = self.history.copy()
            temp_history.append({"role": "user", "parts": [prompt]})
            
            response = self.model.generate_content(temp_history)
            return response.text
        except Exception as e:
            return f"Error generating MoM: {e}"

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
