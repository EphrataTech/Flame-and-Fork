from fastapi import FastAPI,status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import ChatRequest, ChatResponse
from db_handler import logger
from chatbot import assistant

try:
    app = FastAPI(title = "Flame and Fork",description="Flame and Fork Chatbot",version="1.1" )
    logger.info("FastAPI app initialized")
except Exception as e:
    logger.critical(f"Failed to initialize fastAPI app: {e}")
    raise


origins = origins=["http://127.0.0.1:5500",
            "http://localhost:5501",
            "http://127.0.0.1:5501",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "https://flameandfork.com",
            "http://localhost:5000",
            "https://flame-and-fork.netlify.app",
            "*"]
try:
    app.add_middleware(
    CORSMiddleware,
    allow_origins=[origins],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)
except Exception as e:
    logger.error(f"Failed to configure CORS: {e}")
    raise

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response_text = assistant(request.prompt)
        if response_text:
            logger.info("Chatbot returned an answer!")
        return ChatResponse(response=response_text)
    except Exception as e:
        logger.exception("The chatbot failed to respond.")
        raise HTTPException(status_code=500, detail="Chatbot failed to respond.")


if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)