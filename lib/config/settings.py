"""
    Configuration module for environment variables, file paths, database settings, and LLM initialization.

"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Define BASEDIR for the Project
BASEDIR = Path(__file__).parents[2]

# Load environment variables locally
if os.path.isfile(os.path.join(BASEDIR, ".env")):
    load_dotenv(os.path.join(BASEDIR, ".env"), override=True) 


# ------------------------ FILES ------------------------

YOUTUBE_URLS = {
    "PLAN_STRATEGIQUE_RENAULUTION_2021": "https://www.youtube.com/watch?app=desktop&v=EtivAvmDr2Q&t=901s&ab_channel=RenaultGroup",
    "Résultats_financiers_2021": "https://www.youtube.com/watch?v=VfIeaIFSCQA&ab_channel=RenaultGroup",
    "Résultats_financiers_2022": "https://www.youtube.com/watch?v=UWHlyjVtwT8&ab_channel=RenaultGroup",
    "Résultats_financiers_2023": "https://www.youtube.com/watch?v=B57wephix-w&ab_channel=RenaultGroup",
    "Résultats_financiers_2024": "https://www.youtube.com/watch?v=BA5ZOtWfpY0&ab_channel=RenaultGroup",
}

YOUTUBE_TRANSCRIPTS_PATH = os.getenv("YOUTUBE_TRANSCRIPTS_PATH", "data/youtube_transcripts")
DATA_EXTRACTED_PATH = os.getenv("DATA_EXTRACTED_PATH", "data/extracted_data")
PDF_FOLDER = os.getenv("DATA_FOLDER_PATH", "data/raw_pdf_data")
TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR", "transcripts")

LOCAL_FILES = [
    os.path.relpath(os.path.join(BASEDIR, PDF_FOLDER, f), BASEDIR)
    for f in os.listdir(os.path.join(BASEDIR, PDF_FOLDER))
]


# ------------------------ POSTGRES ------------------------

PG_HOST = os.getenv("PG_VECTOR_HOST")
PG_USER = os.getenv("PG_VECTOR_USER")
PG_PASSWORD = os.getenv("PG_VECTOR_PASSWORD")
COLLECTION_NAME = os.getenv("PGDATABASE")

CONNECTION_STRING = f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:5432/{COLLECTION_NAME}"


# ------------------------ LLM  ------------------------

llm_provider = os.getenv("LLM", "OPENAI").upper()

if llm_provider == "OPENAI":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0)
    vision_model = model
elif llm_provider == "GROQ":
    groq_api_key = os.getenv("GROQ_API_KEY")
    model = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192", temperature=0)
    vision_model = ChatGroq(api_key=groq_api_key, model="llama-3.2-90b-vision-preview", temperature=0)
else:
    raise ValueError(f"Unsupported LLM provider: {llm_provider}")


# ------------------------ OTHER CONSTANTS ------------------------

CACHE_DIR = "cache"
CHROMA_PATH = "chroma"
ID_KEY = "doc_id"
LOG_FILE = "youtube_transcripts.log"