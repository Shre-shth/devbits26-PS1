import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

class Brain:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Google API Key is required for Gemini.")

        # Configure the native SDK
        genai.configure(api_key=api_key)
        
        # Use valid model name
        # 🔥 Switch to gemini-2.5-flash as requested
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash-lite',
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
0. whats the name ??
1. Current Location (Where do they stay?), and specify the locality of interest.
2. Requirement (e.g., 3BHK, 4BHK, Villa, Plot)
3. Budget
4. Timeline (When do they plan to move?)

**Contextual Logic:**
- If the user is calling from a region, then suggest properties in that region or nearby regions only.
- If the user asks for a specific type (e.g., "Villa"), do NOT pitch other types (e.g., Apartments) unless asked.
- tailoring your recommendations based on their location and preferences.

**Closing Protocol:**
- If the user indicates they are done (e.g., "That's all", "Bye"), ask: "Is there anything else I can assist you with?"
- If they say "No":
  1. Say a professional goodbye.
  2. Tell them that our team will get back to them as soon as possible.

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

    def update_system_instruction(self, context_text):
        """
        Appends a new system-level instruction to the history to update context.
        """
        print(f"[Brain] Updating system context: {context_text}")
        self.history.append({"role": "user", "parts": [f"SYSTEM UPDATE: {context_text}"]})
        self.history.append({"role": "model", "parts": ["Context updated. I will proceed with this new information."]})

    def generate_mom(self):
        """
        Generates a minutes of the meeting based on the history.
        """
        print("[Brain] Generating Minutes of Meeting...")
        
        # 1. Convert history to a transcript string
        transcript = ""
        for entry in self.history:
            role = entry["role"]
            text = entry["parts"][0]
            
            # Skip system instructions
            if "system_instruction" in text or "SYSTEM UPDATE" in text:
                continue
                
            if role == "user":
                transcript += f"Customer: {text}\n"
            elif role == "model":
                transcript += f"Agent: {text}\n"

        # 2. Create a NEW prompt for the model (forcing a persona switch)
        mom_prompt = f"""
You are an expert Conversation Analyst and Meeting Scribe. Your task is to process the raw transcript of a conversation and generate a structured "Minutes of the Meeting" (MOM) document.

**Objective:**
Analyze the dialogue to extract critical information, specifically focusing on the participants' intent, interests, and the core progression of the discussion.

**Instructions:**
1.  **Analyze Context:** Read the entire conversation history provided.
2.  **Identify Interests:** specifically look for signals regarding what the participants liked, asked about repeatedly, or reacted positively to.
3.  **Extract Key Points:** Summarize the main topics discussed, removing conversational filler (greetings, small talk).
4.  **Determine Status:** Conclude with the final state of the interaction (e.g., issue resolved, follow-up needed, next meeting scheduled).

**Output Format:**
You must provide the output in plain text format without any markdown (no #, *, or bolding). Use the following structure:

1. EXECUTIVE SUMMARY
[Summary here]

2. PARTICIPANT PROFILES & INTERESTS
- Primary Interest: [Interest]
- Sentiment: [Sentiment]
- Key Pain Points/Needs: [Needs]

3. KEY DISCUSSION POINTS
- [Point 1]
- [Point 2]
- [Point 3]

4. ACTION ITEMS / NEXT STEPS
- [Action 1]
- [Action 2]

---
**Input Data:**
{transcript}
"""
        
        try:
            # 3. Send as a fresh request (stateless)
            response = self.model.generate_content(mom_prompt)
            return response.text
        except Exception as e:
            return f"Error generating MoM: {e}"

    def generate_mom_from_transcript(self, transcript: str):
        """
        Generates a minutes of the meeting based on a raw transcript string. for the smart secretary part.
        """
        print("[Brain] Generating Minutes of Meeting from transcript...")
        
        # Create a NEW prompt for the model (forcing a persona switch)
        mom_prompt = f"""
You are an expert Conversation Analyst and Meeting Scribe. Your task is to process the raw transcript of a conversation and generate a structured "Minutes of the Meeting" (MOM) document.

**Objective:**
Analyze the dialogue to extract critical information, specifically focusing on the participants' intent, interests, and the core progression of the discussion.

**Instructions:**
1.  **Analyze Context:** Read the entire conversation history provided.
2.  **Identify Interests:** specifically look for signals regarding what the participants liked, asked about repeatedly, or reacted positively to.
3.  **Extract Key Points:** Summarize the main topics discussed, removing conversational filler (greetings, small talk).
4.  **Determine Status:** Conclude with the final state of the interaction (e.g., issue resolved, follow-up needed, next meeting scheduled).

**Output Format:**
You must provide the output in plain text format without any markdown (no #, *, or bolding). Use the following structure:

1. EXECUTIVE SUMMARY
[Summary here]

2. PARTICIPANT PROFILES & INTERESTS
- Primary Interest: [Interest]
- Sentiment: [Sentiment]
- Key Pain Points/Needs: [Needs]

3. KEY DISCUSSION POINTS
- [Point 1]
- [Point 2]
- [Point 3]

4. ACTION ITEMS / NEXT STEPS
- [Action 1]
- [Action 2]

5. DETAILED SUMMARY
[Detailed summary here] which includes a more detailed analysis of the conversation, including any additional insights.
---
**Input Data:**
{transcript}
"""
        
        try:
            # Send as a fresh request (stateless)
            response = self.model.generate_content(mom_prompt)
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
            import traceback
            error_trace = traceback.format_exc()
            print(f"Brain Error: {e}\n{error_trace}", flush=True)
            # Rollback history on error (remove user message)
            if self.history and self.history[-1]["role"] == "user":
                 self.history.pop()
                 
            yield "not working buddy ! (API Error: " + str(e) + ")"
