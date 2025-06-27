# Flame and Fork API

A FastAPI-powered backend for the **Flame and Fork** website
**Gemini 2.0 Flash** is AI model for intelligent chatbot interactions.

### Deployed API Link
**[flameandfork-api.onrender.com](https://flameandfork-api.onrender.com)**

## Overview
This backend is designed using **FastAPI** to ensure high performance, automatic OpenAPI documentation, and simplicity. It leverages **Google's Gemini 2.0 Flash model** via the `google.generativeai` library to serve AI-generated responses for the application.

##Tech Stack
- **Backend Framework**: FastAPI  
- **AI Integration**: Gemini 2.0 Flash (via `google.generativeai`)
- **Deployment**: Render  
- **Language**: Python 3.10+  
- **API Docs**: `/docs` (Swagger UI) and `/redoc`

##Project Structure
├── app/
│ ├── main.py # FastAPI app entry point
│ ├── chatbot.py # Gemini integration logic
│ ├── models.py # Pydantic models for request/response
│ └── ...
├── requirements.txt # All required Python packages
└── README.md 

##Installation and Setup
- clone the github repository from: git clone https://github.com/your-username/flameandfork-api.git

- Create and activate a virtual environment
- Install dependencies
- Add your API key in a .env file
- Run the app using `uvicorn app.main:app --reload`

##Testing
