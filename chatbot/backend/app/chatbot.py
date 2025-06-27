import google.generativeai as genai
from pathlib import Path
from models import AIPlatform
from dotenv import load_dotenv
import json
import os

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
restaurant_dir = BASE_DIR / "restaurant_details"

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
        self.api_key = api_key
        self.system_prompt = load_system_prompt()
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    
    def chat(self, prompt : str) -> str:
        prompt = f"{self.system_prompt}\n\n{prompt}"
        response = self.model.generate_content(prompt)
        return response.text

api_key = os.getenv("API_KEY") 
