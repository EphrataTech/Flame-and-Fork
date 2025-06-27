from fastapi import FastAPI,status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from chatbot import Gemini, api_key
from models import ChatRequest, ChatResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# App initialization
try:
    app = FastAPI()
except Exception as e:
    logger.critical(f"Failed to initialize fastAPI app: {e}")
# CORS configuration
try:
    app.add_middleware(
    CORSMiddleware,
    # Allows all origins, adjust as needed
    allow_origins=["https://flameandfork.com", "http://localhost:8000","https://flameandfork.onrender.com/"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # Allows all methods, adjust as needed
    allow_headers=["*"],  # Allows all headers, adjust as needed
)
except Exception as e:
    logger.error(f"Failed to configure CORS: {e}")
#Initialize AI platform
try:
    if not api_key:
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


if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)