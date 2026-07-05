"""
template.py

Run this once to scaffold a local RAG project using Ollama (no OpenAI / no API keys).

Usage:
    python template.py
    python template.py --project-name my-rag-project
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

STRUCTURE = [
    "data/",
    "vectorstores/",
    "src/",
    "src/__init__.py",
    "src/pdf_reader.py",
    "src/helper_functions.py",
    "src/retriever.py",
    "src/chains.py",
    "src/config.py",
    "notebooks/",
    "notebooks/evaluation.ipynb",
    "tests/",
    "tests/__init__.py",
    "tests/test_pdf_reader.py",
    ".gitignore",
    "requirements.txt",
    "main.py",
    "README.md",
]

GITIGNORE_CONTENT = """\
venv/
__pycache__/
*.pyc

# Vector stores (rebuild from source PDFs anytime)
vectorstores/

# Jupyter
.ipynb_checkpoints/
"""

REQUIREMENTS_CONTENT = """\
pymupdf
langchain
langchain-community
langchain-ollama
pypdf
faiss-cpu
ollama
"""

CONFIG_CONTENT = """\
# src/config.py
# Central config for the RAG project — edit model names here.

# Ollama base URL (default when running locally)
OLLAMA_BASE_URL = "http://localhost:11434"

# LLM used for generation
LLM_MODEL = "llama3"

# Model used for embeddings
EMBEDDING_MODEL = "nomic-embed-text"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Paths
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstores"
"""

PDF_READER_CONTENT = '''\
# src/pdf_reader.py
import pymupdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from src.config import EMBEDDING_MODEL, OLLAMA_BASE_URL, CHUNK_SIZE, CHUNK_OVERLAP


class PDFReader:
    """Handles PDF ingestion and vector store creation using a local Ollama model."""

    def read_pdf(self, file_path: str) -> str:
        """
        Extracts raw text from a PDF using PyMuPDF.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Full extracted text as a string.
        """
        text = ""
        with pymupdf.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    def encode_pdf(self, path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> FAISS:
        """
        Loads a PDF, splits it into chunks, embeds with Ollama, and returns a FAISS vector store.

        Args:
            path: Path to the PDF file.
            chunk_size: Token size per chunk.
            chunk_overlap: Overlap between consecutive chunks.

        Returns:
            A FAISS vector store.
        """
        loader = PyPDFLoader(path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)

        # Replace tabs with spaces for cleaner embeddings
        for chunk in chunks:
            chunk.page_content = chunk.page_content.replace("\\t", " ")

        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
'''

CHAINS_CONTENT = '''\
# src/chains.py
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

from src.config import LLM_MODEL, OLLAMA_BASE_URL


def build_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    """
    Builds a RetrievalQA chain using a local Ollama LLM.

    Args:
        vectorstore: A FAISS vector store to use as retriever.

    Returns:
        A LangChain RetrievalQA chain.
    """
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )
    return chain
'''

RETRIEVER_CONTENT = '''\
# src/retriever.py
from langchain_community.vectorstores import FAISS


def retrieve_context(question: str, vectorstore: FAISS, k: int = 4) -> list:
    """
    Retrieves the top-k most relevant chunks for a question.

    Args:
        question: The user query.
        vectorstore: The FAISS vector store to search.
        k: Number of chunks to return.

    Returns:
        List of LangChain Document objects.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)


def show_context(docs: list):
    """Prints retrieved chunks for inspection."""
    for i, doc in enumerate(docs, 1):
        print(f"--- Chunk {i} ---")
        print(doc.page_content[:300])
        print()
'''

HELPER_CONTENT = '''\
# src/helper_functions.py
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from src.config import EMBEDDING_MODEL, OLLAMA_BASE_URL, VECTORSTORE_DIR
import os


def save_vectorstore(vectorstore: FAISS, name: str):
    """Persists a FAISS vector store to disk."""
    path = os.path.join(VECTORSTORE_DIR, name)
    vectorstore.save_local(path)
    print(f"Vector store saved to {path}")


def load_vectorstore(name: str) -> FAISS:
    """Loads a persisted FAISS vector store from disk."""
    path = os.path.join(VECTORSTORE_DIR, name)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
'''

MAIN_PY_CONTENT = '''\
# main.py
from src.pdf_reader import PDFReader
from src.chains import build_qa_chain
from src.retriever import show_context
from src.helper_functions import save_vectorstore, load_vectorstore
from src.config import VECTORSTORE_DIR
import os


def main():
    pdf_path = "data/decision-trees.pdf"
    store_name = "decision-trees"
    store_path = os.path.join(VECTORSTORE_DIR, store_name)

    reader = PDFReader()

    # Build or load the vector store
    if os.path.exists(store_path):
        print("Loading existing vector store...")
        vectorstore = load_vectorstore(store_name)
    else:
        print("Building vector store from PDF...")
        vectorstore = reader.encode_pdf(pdf_path)
        save_vectorstore(vectorstore, store_name)

    # Build the QA chain
    chain = build_qa_chain(vectorstore)

    # Ask a question
    question = "What is a decision tree?"
    print(f"\\nQuestion: {question}")
    result = chain.invoke({"query": question})
    print(f"\\nAnswer: {result[\'result\']}")
    print("\\nSource chunks:")
    show_context(result["source_documents"])


if __name__ == "__main__":
    main()
'''

README_CONTENT = """\
# RAG Project (Local — Ollama)

## Structure
- `data/` — source PDFs
- `vectorstores/` — persisted FAISS indexes (gitignored, rebuilt from data/)
- `src/` — core modules: ingestion, retrieval, chains, config
- `notebooks/` — experiments and evaluation
- `tests/` — unit tests
- `main.py` — entry point

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama
```bash
# Install: https://ollama.com
ollama pull llama3
ollama pull nomic-embed-text
```

### 3. Run
```bash
python main.py
```

## Config
Edit `src/config.py` to change the LLM model, embedding model, chunk size, etc.
No API keys needed — everything runs locally via Ollama.
"""

TEST_CONTENT = '''\
# tests/test_pdf_reader.py
import pytest
from src.pdf_reader import PDFReader


def test_read_pdf_returns_string(tmp_path):
    # Place a real PDF in data/ and update the path to run this test
    reader = PDFReader()
    # text = reader.read_pdf("data/decision-trees.pdf")
    # assert isinstance(text, str)
    # assert len(text) > 0
    pass  # placeholder until a test PDF is available
'''


FILE_CONTENTS = {
    ".gitignore": GITIGNORE_CONTENT,
    "requirements.txt": REQUIREMENTS_CONTENT,
    "README.md": README_CONTENT,
    "main.py": MAIN_PY_CONTENT,
    "src/config.py": CONFIG_CONTENT,
    "src/pdf_reader.py": PDF_READER_CONTENT,
    "src/chains.py": CHAINS_CONTENT,
    "src/retriever.py": RETRIEVER_CONTENT,
    "src/helper_functions.py": HELPER_CONTENT,
    "tests/test_pdf_reader.py": TEST_CONTENT,
}


def create_structure(base_dir: Path):
    for entry in STRUCTURE:
        path = base_dir / entry
        if entry.endswith("/"):
            path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                content = FILE_CONTENTS.get(entry, "")
                path.write_text(content, encoding="utf-8")
                logging.info(f"Created file:      {path}")
            else:
                logging.info(f"Skipped (exists):  {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scaffold a local RAG project using Ollama."
    )
    parser.add_argument(
        "--project-name",
        default=".",
        help="Directory to create the project in (default: current directory).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = Path(args.project_name)
    base_dir.mkdir(parents=True, exist_ok=True)
    create_structure(base_dir)
    logging.info(f"Project scaffold created at: {base_dir.resolve()}")