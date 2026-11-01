from Generate import Generate

if __name__ == "__main__":
    question = "How to estimate and error using Lagrange?"
    pdf_path = "files/Interpolation.pdf"
    answer = Generate(question, pdf_path)
    print(answer)