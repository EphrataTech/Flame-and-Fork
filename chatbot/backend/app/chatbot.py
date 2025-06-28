import google.generativeai as genai
from google.generativeai.types import BlockedPromptException
from pathlib import Path
from .models import AIPlatform
from dotenv import load_dotenv
import json
import os
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.getenv("API_KEY")
BASE_DIR = Path(__file__).resolve().parent.parent
restaurant_dir = BASE_DIR / "restaurant_details"
if not restaurant_dir.exists():
    logger.critical(f"Restaurant details directory not found: {restaurant_dir}")
    raise FileNotFoundError(f"Could not find the restaurant details directory :{restaurant_dir}")

def load_system_prompt():
    # Add more files to load data from
    with open(f"{restaurant_dir}/system_prompt.md", "r") as prompt_file:
        prompt_content = prompt_file.read().strip()
    with open(f"{restaurant_dir}/restaurant_menu.json", "r") as menu:
        menu_text = json.dumps(json.load(menu), indent=2)
    with open(f"{restaurant_dir}/restaurant_operations.json", "r") as restaurant_operations:
        operations_content = json.dumps(json.load(restaurant_operations), indent=2)
    with open(f"{restaurant_dir}/restaurant_services.json", "r") as restaurant_services:
        services_content = json.dumps(json.load(restaurant_services),indent=2)
    return f"{prompt_content}\n\n--- MENU ---\n{menu_text}\n\n--- OPERATIONS ---\n{operations_content}\n\n--- SERVICES ---\n{services_content}"

class Gemini(AIPlatform):
    def __init__(self, api_key: str):
        if not api_key:
            logger.critical("API key has not been provided")
            raise ValueError("API key has not been provided")
        self.api_key = api_key.strip()
        self.system_prompt = load_system_prompt()
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            logger.info("Gemini AI has been loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to configure Gemini API : {e}")
            raise RuntimeError(f"Could no configure AI service: {e}")
        
    def chat(self, prompt : str) -> str:
        if not prompt:
            raise ValueError("Prompt is required.")
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")
        if len(prompt.strip()) == 0:
            raise ValueError("Prompt cannot be blank.")
        try:

            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            response = self.model.generate_content(full_prompt)
            if not response:
                raise RuntimeError("Gemini failed to respond correctly.")
            if not response.text:
                raise RuntimeError("AI service returned empty text reponse")
            return response.text
        ##These are errors specifically from Gemini
        except BlockedPromptException as e:
            logger.warning(f"Prompt was blocked: {e}")
            raise ValueError("Prompt blocked by safety filters. Please rephrase.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise RuntimeError(f"Chatbot error: {e}")
