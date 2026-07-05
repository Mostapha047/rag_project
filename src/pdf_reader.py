from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from .config import DATA_DIR ,chunk_size, chunk_overlap, embeddings
from helper_functions import replace_t_with_space

def encode_document(file_path: str):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """
    # Load the PDF document
    loader = PyPDFLoader(file_path)
    document = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = text_splitter.split_documents(document)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings
    get_langchain_embedding_provider(embeddings)
   # Create vector store
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore
