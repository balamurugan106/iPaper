# nlp_utils.py
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
import io



# --- Extract Text from PDF ---
def extract_text_from_pdf(source):
    """
    Accepts either a file path (str) or a file-like object (BytesIO / file stream).
    Returns extracted text (string).
    """
    try:
        if isinstance(source, (str, bytes)):
            # path string given
            reader = PdfReader(source)
        else:
            # file-like
            source.seek(0)
            reader = PdfReader(source)
    except Exception as e:
        # try treating as path
        try:
            reader = PdfReader(str(source))
        except Exception:
            raise

    pages = []
    for p in reader.pages:
        try:
            t = p.extract_text()
        except Exception:
            t = ""
        if t:
            pages.append(t)
    return "\n".join(pages)
    

# --- Simple Extractive Summarizer ---
def simple_sentence_tokenize(text):
    # Splits text into sentences using punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 10]  # remove very short fragments


# --- Topic Clustering (KMeans) ---
def cluster_topics(texts, n_clusters=3):
    if not texts:
        return []
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_
    topics = [f"Topic {i+1}" for i in labels]
    return topics


