from fastapi import FastAPI
from pydantic import BaseModel

from src.chains import ask


class AskRequest(BaseModel):
    question: str
    pdf_path: str


app = FastAPI()


@app.post("/ask")
def ask_endpoint(request: AskRequest):
    answer = ask(request.question, request.pdf_path)
    return {"answer": answer}
