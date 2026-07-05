from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .config import CHUNK_SIZE, CHUNK_OVERLAP, embeddings
from .helper_functions import replace_t_with_space


def encode_document(file_path: str):
    """
    Encodes a PDF into a FAISS vector store using the project's local embedding model.

    Args:
        file_path: The path to the PDF file.

    Returns:
        A FAISS vector store containing the encoded document content.
    """
    # Load the PDF document
    loader = PyPDFLoader(file_path)
    document = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    texts = text_splitter.split_documents(document)
    cleaned_texts = replace_t_with_space(texts)

    # Create vector store
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore
