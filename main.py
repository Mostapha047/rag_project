from src.chains import ask


def main():
    print("RAG system ready. Type 'exit' to quit.")

    pdf_path = input("Path to the PDF you want to ask about: ").strip()

    while True:
        question = input("\nAsk a question: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer = ask(question, pdf_path)
        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
