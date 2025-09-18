# nlp/clustering.py
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

EMB_MODEL_NAME = "all-MiniLM-L6-v2"  # compact & fast (384 dims)
_embed_model = None
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMB_MODEL_NAME)
    return _embed_model

def embed_texts(texts):
    model = get_embed_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def cluster_embeddings(embeddings, n_clusters=6):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(embeddings)
    return labels, km

def extract_cluster_keywords(texts, labels, top_n=6):
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.8)
    X = tfidf.fit_transform(texts)
    terms = tfidf.get_feature_names_out()
    clusters = {}
    for lab in sorted(set(labels)):
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            clusters[lab] = []
            continue
        mean_tfidf = X[idx].mean(axis=0).A1
        top_terms = [terms[i] for i in mean_tfidf.argsort()[-top_n:][::-1]]
        clusters[lab] = top_terms
    return clusters
