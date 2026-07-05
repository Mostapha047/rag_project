from langchain_ollama import chatOllama 
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
embedding_model = chatOllama(model=model, tokenizer=tokenizer, model_path="Qwen/Qwen3-Embedding-0.6B") 
