import fitz  # PyMuPDF
import os


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a single PDF.
    """
    text = ""

    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            text += page.get_text()

        doc.close()

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    return text


if __name__ == "__main__":
    pdf_file = input("Enter PDF path: ")

    if os.path.exists(pdf_file):
        extracted_text = extract_text_from_pdf(pdf_file)

        print("\n========== Extracted Text ==========\n")
        print(extracted_text[:3000])  # Prints first 3000 characters
    else:
        print("File not found.")