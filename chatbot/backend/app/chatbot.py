import google.generativeai as genai
from pathlib import Path
from models import AIPlatform, ChatRequest, ChatResponse
import json

BASE_DIR = Path(__file__).resolve().parent.parent
restaurant_dir = BASE_DIR / "restaurant_details"

def load_system_prompt():
    # Add more files to load data from
    with open(f"{restaurant_dir}/system_prompt.md", "r") as prompt_file:
        prompt_content = prompt_file.read().strip()
    with open(f"{restaurant_dir}/restaurant_menu.json", "r") as menu:
        menu_content = json.load(menu)
    with open(f"{restaurant_dir}/restaurant_operations.json", "r") as restaurant_operations:
        operations_content = json.load(restaurant_operations)
    with open(f"{restaurant_dir}/restaurant_services.json", "r") as restaurant_services:
        services_content = json.load(restaurant_services)
    return f"{prompt_content}\n\n{menu_content}\n\n{operations_content}\n\n{services_content}"

class Gemini(AIPlatform):
    def __init__(self, api_key: str, system_prompt:str = load_system_prompt()):
        self.api_key = api_key
        self.system_prompt = system_prompt
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    
    def chat(self, prompt : str) -> str:
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{prompt}"
        response = self.model.generate_content(prompt)
        return response.text


