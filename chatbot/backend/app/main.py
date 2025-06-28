from fastapi import FastAPI,status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .chatbot import Gemini
from .models import ChatRequest, ChatResponse
from dotenv import load_dotenv
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables
try:
    load_dotenv()
    logger.info("Environment variables loaded")
except Exception as e:
    logger.critical(f"Failed to load .env file: {e}")
    raise RuntimeError("Could not configure the AI service: {e}")
api_key = os.getenv("API_KEY")
#App initialization
try:
    app = FastAPI()
    logger.info("FastAPI app initialized")
except Exception as e:
    logger.critical(f"Failed to initialize fastAPI app: {e}")
    raise

# CORS configuration
try:
    app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500",
            "http://localhost:5501",
            "http://127.0.0.1:5501",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "https://flameandfork.com",
            "http://localhost:5000",
            "https://flameandfork.onrender.com",
            "https://flame-and-fork.netlify.app",
            "null"],
    allow_credentials=False,
    allow_methods=["POST", "GET","OPTIONS"],
    allow_headers=["*"],  # Allows all headers
)
except Exception as e:
    logger.error(f"Failed to configure CORS: {e}")
    raise
#Initialize AI platform

try:
    if not api_key:
        logger.critical("Could not get the env variable")
        raise ValueError("API key is not available")
    ai_platform = Gemini(api_key)
    logger.info("AI platform started successfully")
except Exception as e:
    logger.critical(f"Failed to initialize AI platform: {e}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API"}

@app.post("/chat", response_model=ChatResponse)
async def chatbot(request: ChatRequest):
    try:
        response_text = ai_platform.chat(request.prompt)
        return ChatResponse(response=response_text)
    except RuntimeError as e:
        logger.error(f"The chatbot has encountered an error: {e}")
        raise HTTPException(status_code=503, detail="Chatbot temporarily unavailable. Please try again later.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected server error.")
    #return ChatResponse(response={request.prompt}")


if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)