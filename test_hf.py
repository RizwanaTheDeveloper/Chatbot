from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    token=os.getenv("HUGGINGFACE_TOKEN"),
    base_url="https://router.huggingface.co/hf-inference"
)

try:
    response = client.text_generation(
        "Hello, how are you?",
        model=os.getenv("MODEL_NAME", "google/flan-t5-small"),
        max_new_tokens=50
    )
    print("✅ Success! Bot says:", response)
except Exception as e:
    print("❌ Error:", e)