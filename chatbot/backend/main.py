from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from abc import ABC, abstractmethod
import os
import google.generativeai as genai

#Load environment variables from .env file
load_dotenv()
#Initialize application
app = FastAPI()

app.add_middleware(CORSMiddleware,
    allow_origins = ["https://flameandfork.com", "http://localhost:8000","https://flameandfork.onrender.com/"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # Allows all methods, adjust as needed
    allow_headers=["*"],  # Allows all headers, adjust as needed)
)

class AIPlatform(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> str:
        pass

@app.get("/")
async def root():
    return {"Message" : "Welcome to the Flame and Fork Chatbot API!"}

@app.post("/chat")
# async def chatbot():
