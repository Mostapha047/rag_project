from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    pdf_path: str
# This works:

req = AskRequest(question="hello", pdf_path="data/decision-trees.pdf")
print(req.question)   # "hello"

# This raises a pydantic.ValidationError:
bad = AskRequest(pdf_path="data/decision-trees.pdf")   # missing "question"