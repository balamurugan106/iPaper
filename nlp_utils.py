import io
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def simple_sentence_tokenize(text):
    """
    Simple sentence splitter (no NLTK). Keeps sentences > ~10 chars.
    """
    if not text:
        return []
    # split on punctuation + whitespace
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    # filter out tiny fragments
    return [s.strip() for s in sents if len(s.strip()) > 10]

def summarize_text(text, num_sentences=3):
    
    sentences = simple_sentence_tokenize(text)
    if not sentences:
        return ""
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vect = TfidfVectorizer(stop_words='english', max_df=0.9)
    X = vect.fit_transform(sentences)
    scores = X.sum(axis=1).A1
    top_idx = scores.argsort()[-num_sentences:]
    top_idx_sorted = sorted(top_idx)
    summary = " ".join([sentences[i] for i in top_idx_sorted])
    return summary

def extract_keywords(text, top_n=5):
    # Clean text
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    # Tokenize and count word frequencies
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    words = vectorizer.get_feature_names_out()
    counts = np.asarray(X.sum(axis=0)).flatten()

    # Sort by frequency
    sorted_indices = counts.argsort()[::-1]
    top_keywords = [words[i] for i in sorted_indices[:top_n]]

    return top_keywords

def cluster_topics(docs, k=3):
    
    if not docs:
        return []
    vect = TfidfVectorizer(stop_words='english', max_df=0.9)
    X = vect.fit_transform(docs)
    n_clusters = min(k, len(docs))
    if n_clusters <= 1:
        return ["Topic 1"] * len(docs)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = model.fit_predict(X)
    return [f"Topic {l+1}" for l in labels]

