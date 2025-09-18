class UserDocument(db.Model):
    __tablename__ = 'user_documents'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String)
    file_path = db.Column(db.String)
    # new fields:
    summary = db.Column(db.Text)
    summary_status = db.Column(db.String(20), default='pending')
    cluster_id = db.Column(db.Integer)
    topics = db.Column(db.Text)
    # if using JSON
    embedding = db.Column(db.JSON)  # or use pgvector.Vector if you installed pgvector support
