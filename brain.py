import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

class Brain:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Google API Key is required for Gemini.")

        # Using valid alias
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest", 
            temperature=0.7, 
            google_api_key=api_key
        )
        
        # Store conversation history
        self.history = []
        
        # System prompt
        self.system_prompt = HumanMessage(content="""You are a helpful customer service voice bot.
Keep your responses concise (1-2 sentences).
Do not use markdown symbols like * or #.
If interrupted, stop immediately.
""")

    def generate_response_stream(self, text):
        """
        Yields text chunks from Gemini.
        """
        messages = [self.system_prompt] + self.history + [HumanMessage(content=text)]
        
        full_response = ""
        try:
            print(f"[Brain] Requesting stream for: '{text}'")
            # invoke streaming
            # Langchain's .stream() is blocking iterator
            for chunk in self.llm.stream(messages):
                content = chunk.content
                text_part = ""
                
                if isinstance(content, str):
                    text_part = content
                elif isinstance(content, list):
                    # Handle structured content [{'type': 'text', 'text': '...'}]
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            text_part += part.get('text', '')
                
                if text_part:
                    print(f"[Brain] Chunk: {text_part}", flush=True)
                    full_response += text_part
                    yield text_part
        except Exception as e:
            print(f"Brain Error: {e}", flush=True)
            yield " I'm having trouble thinking."
            
        finally:
            # Update history even if interrupted
            self.history.append(HumanMessage(content=text))
            self.history.append(AIMessage(content=full_response))
