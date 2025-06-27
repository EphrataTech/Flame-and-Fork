from fastapi import FastAPI,status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from chatbot import Gemini, api_key
from models import ChatRequest, ChatResponse
# App initialization
app = FastAPI()
# CORS configuration
app.add_middleware(
    CORSMiddleware,
    # Allows all origins, adjust as needed
    allow_origins=["https://flameandfork.com", "http://localhost:8000","https://flameandfork.onrender.com/"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # Allows all methods, adjust as needed
    allow_headers=["*"],  # Allows all headers, adjust as needed
)

ai_platform = Gemini(api_key)

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API"}

@app.post("/chat", response_model=ChatResponse)
async def chatbot(request: ChatRequest):
    response_text = ai_platform.chat(request.prompt)
    return ChatResponse(response=response_text)

if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)