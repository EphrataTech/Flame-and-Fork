from pydantic import BaseModel
from abc import ABC, abstractmethod
from chatbot import load_system_prompt

class AIPlatform(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> str:
        """Method to handle chat requests"""
        pass

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str  

