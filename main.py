from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
from dotenv import load_dotenv
import uvicorn
from logging.handlers import RotatingFileHandler
from google import generativeai as genai
from google.generativeai import types

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGIN"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to initialize Gemini")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 2048

class ChatResponse(BaseModel):
    response: str
    error: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_with_genai(request: ChatRequest):
    try:
        # Validate messages
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")

        # Create a chat session
        chat = model.start_chat(history=[])

        # Format messages for the chat
        for msg in request.messages:
            if msg.role.lower() == "user":
                chat.send_message(msg.content)

        # Get the last message from the user
        last_message = request.messages[-1].content

        # Generate response
        # logger.info("Calling Gemini API...")
        response = model.generate_content(
            last_message,
            generation_config=types.GenerationConfig(
                max_output_tokens=request.max_tokens,
                temperature=0.7
            )
        )

        # Check if the request was successful
        if not response:
            logger.error("Empty response from Gemini API")
            raise HTTPException(status_code=503, detail="Service unavailable")

        # Get the response text
        result = response.text
        return ChatResponse(
            response=result,
            error=None
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return ChatResponse(
            response="",
            error=str(e)
        )

@app.get("/health")
async def health_check():
    try:
        # Simple test generation to verify API is working
        response = model.generate_content("Test")
        return {
            "status": "healthy",
            "api_connected": True,
            "test_generation_successful": bool(response and response.text)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
