from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstores"

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "BAAI/bge-small-en"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
