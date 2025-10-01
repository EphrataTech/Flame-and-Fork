from langchain_community.document_loaders import JSONLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from pathlib import Path
from dotenv import load_dotenv
import logging
import os

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
restaurant_dir = BASE_DIR / "restaurant_details"
if not restaurant_dir.exists():
    logger.critical(f"Restaurant details directory not found: {restaurant_dir}")
    raise FileNotFoundError(f"Could not find the restaurant details directory :{restaurant_dir}")

persist_directory = BASE_DIR/"db"
json_loader = DirectoryLoader(
    str(restaurant_dir),
    loader_cls=lambda file_path: JSONLoader(
        file_path=file_path,
        jq_schema=".",
        text_content=False,),
        glob="*.json")
try:
    json_docs = json_loader.load()
    logger.info("Successfully loaded the JSON files")
except Exception as e:
    logger.exception("Error in loading the JSON files!")

try:
    embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    logger.info("Successfully loaded Gemini model")
except Exception as e:
    logger.exception("Failed to load the embedding model")

try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
    logger.info("Initiated the text splitter")
except Exception as e:
    logger.exception("Failed to initialize the text splitter")

split_text = text_splitter.split_documents(json_docs)

# Populating the database
try:

    vectordb = Chroma.from_documents(
        persist_directory=persist_directory,
        embedding=embeddings,
        documents=split_text
    )
except Exception as e:
    logger.exception("Failed to populate the Database")

# Retrieving the database
try:
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever()
    logger.info("Successfully loaded the Chroma vector database")
except Exception as e:
    logger.exception("Failed to load the Chroma vector database")