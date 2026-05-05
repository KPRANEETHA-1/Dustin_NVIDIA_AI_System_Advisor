from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# List available models on your key
print("Available models:")
for model in client.models.list():
    print(f"  {model.name}")