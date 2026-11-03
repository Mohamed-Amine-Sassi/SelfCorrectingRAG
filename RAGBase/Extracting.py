from pypdf import PdfReader

def Extracting(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(f"Extracted text from {pdf_path}.")
    return text

