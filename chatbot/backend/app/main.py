from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


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