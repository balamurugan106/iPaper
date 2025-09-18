# routes/routes_nlp.py
from flask import Blueprint, jsonify, request, current_app
from your_app import db  # adjust import to your app
from models import UserDocument, User  # adjust to your models
from nlp.nlp_helpers import extract_text
from nlp.summarizer import summarize_long_text, extractive_summary
from nlp.clustering import embed_texts
import numpy as np

nlp_bp = Blueprint('nlp', __name__, url_prefix='/nlp')

def current_user():
    # replace with your login/session function
    from flask_login import current_user
    return current_user

@nlp_bp.route('/generate_summary/<int:doc_id>', methods=['POST'])
def generate_summary(doc_id):
    user = current_user()
    if not user.is_authenticated:
        return jsonify({"error":"login required"}), 401

    if getattr(user, 'membership', 'Free') == 'Free':
        return jsonify({"error":"Upgrade required"}), 403

    doc = UserDocument.query.get(doc_id)
    if not doc:
        return jsonify({"error":"Document not found"}), 404

    doc.summary_status = 'processing'
    db.session.commit()

    try:
        text = extract_text(doc.file_path)
    except Exception as e:
        doc.summary_status = 'error'
        db.session.commit()
        return jsonify({"error":"text extraction failed: "+str(e)}), 500

    try:
        summary = summarize_long_text(text)
    except Exception as e:
        summary = extractive_summary(text)

    # save results
    doc.summary = summary
    doc.summary_status = 'done'
    # compute and save embedding (store as JSON list) - optional better: pgvector
    try:
        emb = embed_texts([summary])[0]
        doc.embedding = emb.tolist()  # ensure your model supports JSON storage or use pgvector
    except Exception as e:
        current_app.logger.warning("Embedding failed: %s", e)

    db.session.commit()
    return jsonify({"summary": summary}), 200

@nlp_bp.route('/status/<int:doc_id>', methods=['GET'])
def status(doc_id):
    doc = UserDocument.query.get(doc_id)
    if not doc:
        return jsonify({"error":"Document not found"}), 404
    return jsonify({
        "summary_status": doc.summary_status,
        "has_summary": bool(doc.summary)
    })

@nlp_bp.route('/cluster_all', methods=['POST'])
def cluster_all():
    # admin check - replace with your admin logic
    user = current_user()
    if not getattr(user, 'is_admin', False):
        return jsonify({"error":"admin only"}), 403

    n_clusters = int(request.json.get('n_clusters', 6))
    docs = UserDocument.query.filter(UserDocument.summary_status == 'done').all()
    texts = [d.summary or "" for d in docs]
    if not texts:
        return jsonify({"error":"no summaries to cluster"}), 400

    embeddings = embed_texts(texts)
    from nlp.clustering import cluster_embeddings, extract_cluster_keywords
    labels, km = cluster_embeddings(embeddings, n_clusters=n_clusters)
    clusters = extract_cluster_keywords(texts, labels, top_n=6)

    for d, lab in zip(docs, labels):
        d.cluster_id = int(lab)
        d.topics = ",".join(clusters[int(lab)])  # store comma separated
    db.session.commit()
    return jsonify({"clusters": clusters}), 200

@nlp_bp.route('/similar/<int:doc_id>', methods=['GET'])
def similar(doc_id):
    k = int(request.args.get('k', 5))
    doc0 = UserDocument.query.get(doc_id)
    if not doc0 or not doc0.embedding:
        return jsonify({"error":"no embedding"}), 400
    emb0 = np.array(doc0.embedding)
    all_docs = UserDocument.query.filter(UserDocument.id != doc_id, UserDocument.embedding.isnot(None)).all()
    sims = []
    for d in all_docs:
        emb = np.array(d.embedding)
        cos = float(np.dot(emb0, emb) / (np.linalg.norm(emb0)*np.linalg.norm(emb)+1e-9))
        sims.append((cos, d.id, d.title))
    sims.sort(reverse=True)
    top = [{"id":sid, "title":stitle, "score":float(score)} for score,sid,stitle in sims[:k]]
    return jsonify({"similar": top})
