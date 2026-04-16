from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(token=os.getenv("HUGGINGFACE_TOKEN"))
model = os.getenv("MODEL_NAME", "google/flan-t5-small")

try:
    response = client.text_generation(
        "Say hello",
        model=model,
        max_new_tokens=20
    )
    print("✅ Working! Response:", response)
except Exception as e:
    print("❌ Error:", e)