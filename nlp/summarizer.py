# nlp/summarizer.py
import re
import logging
from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logger = logging.getLogger(__name__)

# model choices that are reasonably small
SUM_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"  # small and decent
_device = 0 if torch.cuda.is_available() else -1

# instantiate lazily to reduce startup time
_summarizer = None
def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model=SUM_MODEL_NAME, device=_device)
    return _summarizer

def sentence_split(text: str):
    return re.split(r'(?<=[.!?])\s+', text)

def chunk_text(sentences: List[str], max_chars: int = 1000):
    chunks = []
    cur, cur_len = [], 0
    for s in sentences:
        slen = len(s)
        if cur_len + slen <= max_chars:
            cur.append(s)
            cur_len += slen
        else:
            chunks.append(" ".join(cur))
            cur = [s]
            cur_len = slen
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def summarize_long_text(text: str, max_length=200, min_length=30):
    sents = sentence_split(text)
    chunks = chunk_text(sents, max_chars=1000)
    summaries = []
    summarizer = get_summarizer()
    for chunk in chunks:
        try:
            out = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(out[0]['summary_text'])
        except Exception as e:
            logger.exception("Summarizer chunk failed, falling back to extractive: %s", e)
            from .summarizer_fallback import extractive_summary
            return extractive_summary(text)
    return " ".join(summaries)

# Fallback simple extractive summarizer (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extractive_summary(text: str, n_sentences=5):
    sents = sentence_split(text)
    if len(sents) <= n_sentences:
        return " ".join(sents)
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(sents)
    scores = X.sum(axis=1).A1
    top_idx = np.argsort(scores)[-n_sentences:][::-1]
    top_idx.sort()
    return " ".join([sents[i] for i in top_idx])
