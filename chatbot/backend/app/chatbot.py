from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from db_handler import logger
import os

try:
    load_dotenv()
    logger.info("Environment variables loaded")
except Exception as e:
    logger.critical(f"Failed to load .env file: {e}")
    raise RuntimeError("Could not configure the AI service: {e}")


try:
    ai_model = os.getenv("GROQ_MODEL")
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=ai_model,api_key=groq_api_key,temperature=0.7, reasoning_effort="medium")
    logger.info("Successfully initiated the AI model")
except Exception as e:
    logger.exception("Failed to initialize the AI model")

chat_prompt_template = ChatPromptTemplate.from_template(
    """
You are an AI Chatbot for flame and fork restaurant. Your role is to intelligently answer the user's query using a 
"""
)

answer_chain = chat_prompt_template | llm | StrOutputParser 