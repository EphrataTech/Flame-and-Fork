from pydantic import BaseModel
from abc import ABC, abstractmethod

class AIPlatform(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> str:
        """Method to handle chat requests"""
        pass

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str  

