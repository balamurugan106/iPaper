import re
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline

# Load transformer model once (cached)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ---------- 1. Extract Text Safely ----------
def extract_text_from_pdf(file_path):
    """
    Safely extract text from PDF, skipping unreadable pages.
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                print(f"⚠️ Skipping page {i+1}: {e}")
        return text.strip() if text.strip() else "No extractable text found."
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return "Error reading PDF file."

# ---------- 2. Split into chunks ----------
def chunk_text(text, max_length=2000):
    """
    Split long text into manageable chunks for summarization.
    """
    for i in range(0, len(text), max_length):
        yield text[i:i + max_length]

# ---------- 3. Summarize ----------
def summarize_text(text):
    """
    Summarize long text in chunks and merge results.
    """
    if not text or len(text) < 100:
        return "Not enough content to summarize."

    summaries = []
    for chunk in chunk_text(text):
        try:
            summary = summarizer(
                chunk,
                max_length=200,
                min_length=60,
                do_sample=False
            )[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"⚠️ Summarization chunk failed: {e}")
            continue

    if not summaries:
        return "Summarization failed."
    return " ".join(summaries)

# ---------- 4. Extract Keywords ----------
def extract_keywords(text, top_n=5):
    """
    Simple keyword extraction using word frequency.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    words = vectorizer.get_feature_names_out()
    counts = np.asarray(X.sum(axis=0)).flatten()
    sorted_indices = counts.argsort()[::-1]
    return [words[i] for i in sorted_indices[:top_n]]
