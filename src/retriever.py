from pathlib import Path

from langchain_community.vectorstores import FAISS

from .config import VECTORSTORE_DIR, embeddings
from .pdf_reader import encode_document


def _vectorstore_path(pdf_path: str) -> Path:
    """Each PDF gets its own vector store folder, named after the PDF itself."""
    return VECTORSTORE_DIR / Path(pdf_path).stem


def create_retriever_from_pdf(pdf_path: str):
    """
    Builds a vector store from a PDF, saves it to disk, and returns a retriever.

    Args:
        pdf_path: The path to the PDF file.
    """
    chunks_vector_store = encode_document(pdf_path)
    chunks_vector_store.save_local(str(_vectorstore_path(pdf_path)))
    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})
    return chunks_query_retriever


def get_retriever(pdf_path: str):
    """
    Loads the saved vector store for this specific PDF if one already exists,
    otherwise builds one from the PDF.

    Args:
        pdf_path: The path to the PDF file.
    """
    store_path = _vectorstore_path(pdf_path)
    index_file = store_path / "index.faiss"
    if index_file.exists():
        chunks_vector_store = FAISS.load_local(
            str(store_path), embeddings, allow_dangerous_deserialization=True
        )
        chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})
    else:
        chunks_query_retriever = create_retriever_from_pdf(pdf_path)
    return chunks_query_retriever
