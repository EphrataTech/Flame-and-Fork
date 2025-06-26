import google.generativeai as genai
from pathlib import Path
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

