from typing import List
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import PyPDF2
import whisper

def load_data_from_url(url: str) -> str:
    """Extract text from a URL using Playwright and BeautifulSoup."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            html = page.content()
            browser.close()
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            return text
    except Exception as e:
        print(f"Error loading data from URL: {e}")
        return ""

def load_data_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error loading data from PDF: {e}")
        return ""

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio from a file using Whisper."""
    try:
        model = whisper.load_model("base")  # You can choose different model sizes
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

# Example usage (for testing):
if __name__ == "__main__":
    # URL example
    url = "https://www.example.com"  # Replace with a real URL
    url_text = load_data_from_url(url)
    print(f"Text from URL: {url_text[:200]}...")

    # PDF example (replace with a real PDF path)
    pdf_path = "example.pdf"  # Create a dummy pdf file
    with open(pdf_path, "w") as f:
        f.write("Dummy PDF content")
    pdf_text = load_data_from_pdf(pdf_path)
    import os
    os.remove(pdf_path)
    print(f"Text from PDF: {pdf_text[:200]}...")

    # Audio example (replace with a real audio path)
    audio_path = "example.mp3" # Create a dummy mp3 file
    with open(audio_path, "w") as f:
        f.write("Dummy mp3 content")
    audio_text = transcribe_audio(audio_path)
    os.remove(audio_path)
    print(f"Text from Audio: {audio_text[:200]}...")

    # Test loading data from the provided PDF path
    pdf_path = "C:\\Users\\hp\\Downloads\\R\\Quantum_Innovations_IT_Security_Policy.pdf"
    pdf_text = load_data_from_pdf(pdf_path)
    print(f"Text from PDF: {pdf_text[:200]}...")
