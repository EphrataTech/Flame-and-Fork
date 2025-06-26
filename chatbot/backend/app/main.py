from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
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


class AIPlatform(ABC):

    @abstractmethod
    def chat(self, prompt: str) -> str:
        """Method to handle chat requests"""
        pass


def load_system_prompt():
    # Add more files to load data from
    with open("restaurant/system_prompt.md", "r") as prompt_file:
        prompt_content = prompt_file.read().strip()
    with open("restaurant/restaurant_menu.json", "r") as menu:
        menu_content = menu.read().strip()
    with open("restaurant/restaurant_operations.json", "r") as restaurant_operations:
        operations_content = restaurant_operations.read().strip()
    with open("restaurant/restaurant_services.json", "r") as restaurant_services:
        services_content = restaurant_services.read().strip()
    return f"{prompt_content}\n\n{menu_content}\n\n{operations_content}\n\n{services_content}"


class Gemini(AIPlatform):
    def __init__(self,api_key,system_prompt: str = "You are a helpful assistant for a restaurant."):
        self.api_key = api_key
        self.system_prompt = system_prompt
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash")

    def chat(self, prompt: str) -> str:
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n {prompt}"
        response = self.model.generate_content(prompt)
        return response.text


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str


api_key = os.getenv("API_KEY")
system_prompt = load_system_prompt()

ai_platform = Gemini(api_key=api_key, system_prompt=system_prompt)


@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API"}


@app.post("/chat", response_model=ChatResponse)
async def chatbot(request: ChatRequest):
    response_text = ai_platform.chat(request.prompt)
    return ChatResponse(response=response_text)

#Import from .env
#Structuring code in what files? 
#CORS
#Error handling
#Docker containerization
#Live public deployment
#Testing with postman
##Frontend connection
##Bonus-combining information from multiple files