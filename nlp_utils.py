import io
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def extract_text_from_pdf(source):
    
  
    try:
        if hasattr(source, "read"):
            source.seek(0)
            reader = PdfReader(source)
        else:
            reader = PdfReader(str(source))
    except Exception:
        # fallback: try path
        reader = PdfReader(str(source))

    pages = []
    for p in reader.pages:
        try:
            t = p.extract_text()
        except Exception:
            t = ""
        if t:
            pages.append(t)
    return "\n".join(pages)

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
   
    if not text:
        return []

    vect = TfidfVectorizer(stop_words='english', max_df=0.9, ngram_range=(1,2))
    X = vect.fit_transform([text])
    feature_names = vect.get_feature_names_out()
    scores = X.toarray()[0]
    if scores.sum() == 0:
        return []
    top_idx = scores.argsort()[-top_n:][::-1]
    keywords = [feature_names[i] for i in top_idx if i < len(feature_names)]
    return keywords

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
