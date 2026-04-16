import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from groq import AsyncGroq  # <-- Import the Groq library
import traceback

load_dotenv()

# ===== CONFIGURATION =====
MONGO_URL = os.getenv("MONGO_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # <-- Load the Groq API key
MODEL_NAME = os.getenv("MODEL_NAME") # <-- Your chosen Groq model

# ===== FASTAPI APP =====
app = FastAPI(title="ChatBot")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ===== MONGODB =====
try:
    client = AsyncIOMotorClient(MONGO_URL)
    db = client.chatbot_db
    collection = db.chatbot_history
    print("MongoDB connected")
except Exception as e:
    print(f"MongoDB error: {e}")
    collection = None

# ===== GROQ CLIENT =====
groq_client = AsyncGroq(api_key=GROQ_API_KEY)  # <-- Initialize the Groq client

logging.basicConfig(level=logging.INFO)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    saved: bool

# ===== AI MODEL REPLY (USING GROQ) =====
async def get_model_reply(user_message: str) -> str:
    if not GROQ_API_KEY:
        return "ERROR: Groq API key is missing."

    try:
        print(f"Calling Groq model: {MODEL_NAME}")
        # <-- Call Groq's chat completion endpoint
        response = await groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=500,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        print(f"Model replied: {reply[:100]}...")
        return reply
    except Exception as e:
        print(f"Groq API error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return f"Model error: {str(e)[:200]}"

# ===== DATABASE FUNCTIONS (UNCHANGED) =====
async def save_message(role: str, content: str) -> bool:
    if collection is None:
        print("No database connection, skipping save")
        return False
    try:
        doc = {"role": role, "content": content, "timestamp": datetime.utcnow()}
        await collection.insert_one(doc)
        print(f"Saved {role} message")
        return True
    except Exception as e:
        print(f"Save failed: {e}")
        return False

async def get_recent_messages(limit: int = 50):
    if collection is None:
        return []
    try:
        cursor = collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
        messages = await cursor.to_list(length=limit)
        return list(reversed(messages))
    except Exception as e:
        print(f"Failed to get messages: {e}")
        return []

# ===== API ENDPOINTS (UNCHANGED) =====
@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_req: ChatRequest):
    user_msg = chat_req.message.strip()
    if not user_msg:
        raise HTTPException(400, "Empty message")
    
    print(f" Received: {user_msg}")
    
    user_saved = await save_message("user", user_msg)
    bot_reply = await get_model_reply(user_msg)
    bot_saved = await save_message("assistant", bot_reply)
    
    return ChatResponse(reply=bot_reply, saved=user_saved and bot_saved)

@app.get("/history")
async def history():
    return await get_recent_messages()

@app.delete("/history")
async def clear():
    if collection is not None:
        await collection.delete_many({})
    return {"status": "cleared"}

@app.get("/db-stats")
async def stats():
    if collection is None:
        return {"connected": False, "error": "Not connected"}
    try:
        count = await collection.count_documents({})
        return {"connected": True, "total_messages": count}
    except Exception as e:
        return {"connected": False, "error": str(e)}