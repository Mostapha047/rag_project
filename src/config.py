from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatOllama(model="qwen3:8b")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)