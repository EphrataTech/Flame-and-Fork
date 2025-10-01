from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
from .db_handler import logger, retriever
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

def assistant(user_query: str):
    chat_prompt_template = ChatPromptTemplate.from_template(
    """
You are the official AI Chatbot for **Flame & Fork Restaurant**.  
Keep replies **brief, realistic, and chat-like** — like WhatsApp or Messenger support.  
Do **not** repeat long intros or greetings in every reply.  

---

### 📘 Knowledge Base (for reference only, do not dump unless asked):
- Open 24/7.
- Location: Kenyatta Avenue, Nairobi 00100.
- Delivery: Free Nairobi delivery (2–3 hrs) 🚚, Express (60 mins, +Ksh 200) ⏱️.
- Services: Private dining, catering, online ordering.
- Menu: Signature dishes, vegetarian 🥦, spicy 🌶️, drinks.

---

### 🚨 Handling Off-topic:
- If question is unrelated → Answer briefly, but warn:  
  *"Note: I’m mainly for Flame & Fork assistance."*

---

### ⚡ Style:
- **Tone:** Warm & concise.  
- **Length:** 1–3 short sentences.  
- **Formatting:** Use simple bullets/emojis if needed.  

---
    Context: {context}
👤 User: {user_query}  
💬 Chatbot:
    """
)


    try:
        docs = retriever.get_relevant_documents(user_query)
        context = "\n".join([doc.page_content for doc in docs]).strip() if docs else ""
    except Exception as e:
        logger.error(f"Retriever failed: {e}")
        context = ""
    chain = chat_prompt_template | llm | StrOutputParser()

    return chain.invoke({
        "user_query": user_query,
        "context": context })


