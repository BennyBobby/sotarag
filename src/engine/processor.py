import fitz
import requests
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter


def download_and_split_pdf(url: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Download a PDF from a URL, extract its text, and split it into chunks.

    Args:
        url (str): The direct link to the PDF file.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        list: A list of text strings (chunks).
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        pdf_stream = io.BytesIO(response.content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        if not full_text.strip():
            print(f"Warning: No text extracted from {url}. PDF might be image-based.")
            return []
        # RecursiveCharacterTextSplitter tries to split on paragraphs, then sentences
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_text(full_text)
        return chunks

    except Exception as e:
        print(f"Error processing PDF at {url}: {e}")
        return []


if __name__ == "__main__":
    test_pdf = "https://arxiv.org/pdf/1706.03762.pdf"
    extracted_chunks = download_and_split_pdf(test_pdf)

    if extracted_chunks:
        print(f"Successfully split into {len(extracted_chunks)} chunks.")
