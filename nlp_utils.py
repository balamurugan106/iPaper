
import os
import time
import math
import re
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from google import genai
from google.genai import types

# initialize client (Gemini Developer API)
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in environment")

client = genai.Client(api_key=API_KEY)   # Gem API key usage (Gemini Developer API). :contentReference[oaicite:6]{index=6}

# ---------- safe PDF extraction ----------
def extract_text_from_pdf(file_path):
    """Extract text safely from a PDF, skipping unreadable pages."""
    try:
        reader = PdfReader(file_path)
        parts = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
                if txt.strip():
                    parts.append(txt)
            except Exception as e:
                # skip problematic pages
                print(f"Skipping page {i+1}: {e}")
        text = "\n".join(parts).strip()
        if not text:
            return ""
        return text
    except Exception as e:
        print("PDF read error:", e)
        return ""

# ---------- chunking helper ----------
def chunk_text(text, max_chars=2500):
    """Yield chunks of about max_chars characters (avoid cutting sentences naively)."""
    if not text:
        yield ""
        return
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            yield text[start:].strip()
            break
        # extend end to nearest sentence boundary (period/newline) up to +200 chars
        tail = text[end:end+200]
        m = re.search(r'[\.!\?]\s', tail)
        if m:
            end = end + m.end()
        yield text[start:end].strip()
        start = end

# ---------- Gemini summarization for one chunk ----------
def summarize_chunk_with_gemini(chunk, model="gemini-2.5-flash", max_retries=2):
    """
    Send one chunk to Gemini and return the summary text.
    We wrap with simple retries/backoff.
    """
    if not chunk:
        return ""

    prompt = (
        "Please produce a concise summary of the following document chunk. "
        "Format as a short paragraph (3-6 sentences). Be factual and preserve main points.\n\n"
        f"{chunk}"
    )

    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt
            )
            # response.text contains the generated text
            return resp.text.strip()
        except Exception as e:
            print(f"Gemini call failed (attempt {attempt}): {e}")
            time.sleep(1 + attempt * 2)
            continue
    return "Summarization failed (downstream API error)."

# ---------- top-level summarizer: chunk, summarize, merge ----------
def summarize_text_with_gemini(text, chunk_max_chars=2500):
    """
    Summarize long text by chunking for long context, summarizing each chunk,
    then merging chunk summaries into a final short summary.
    """
    if not text or len(text.strip()) < 50:
        return "Not enough extractable content to summarize."

    chunk_summaries = []
    for chunk in chunk_text(text, max_chars=chunk_max_chars):
        s = summarize_chunk_with_gemini(chunk)
        chunk_summaries.append(s)

    # If only one chunk, return it; otherwise combine and ask Gemini to condense
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined = "\n\n".join(chunk_summaries)
    # Ask Gemini to combine summaries into final concise summary
    final_prompt = (
        "You are given several intermediate summaries (each from a chunk of the same document). "
        "Please combine them into a single concise summary (4-6 sentences) that covers the main points:\n\n"
        f"{combined}"
    )
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=final_prompt)
        return resp.text.strip()
    except Exception as e:
        print("Final combine failed:", e)
        # fallback: join chunk summaries
        return " ".join(chunk_summaries)

# ---------- keywords (simple freq-based) ----------
def extract_keywords(text, top_n=6):
    if not text:
        return []
    txt = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([txt])
    words = vectorizer.get_feature_names_out()
    counts = np.asarray(X.sum(axis=0)).flatten()
    idxs = counts.argsort()[::-1][:top_n]
    return [words[i] for i in idxs]
