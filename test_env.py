import os
from dotenv import load_dotenv

# Force reload
load_dotenv(override=True)

api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    print(f"Loaded Key: {api_key[:6]}...{api_key[-4:]}")
else:
    print("No key loaded.")
