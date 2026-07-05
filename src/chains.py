from .config import llm
from .retriever import get_retriever


def ask(question: str, pdf_path: str) -> str:
    retriever = get_retriever(pdf_path)
    docs = retriever.invoke(question)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a helpful assistant.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return str(response.content)
