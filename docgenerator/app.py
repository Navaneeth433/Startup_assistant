import os
from flask import Flask, render_template, request, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from DocsGenerator.generator import generate_nda, generate_pitch_deck, generate_mou, generate_rti

app = Flask(__name__)

# -------------------
# Database Config
# -------------------
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:300234@localhost:5432/startup_assistant"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# -------------------
# Models (simplified, import yours in real usage)
# -------------------
class User(db.Model):
    __tablename__ = "users"
    user_id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String)
    email = db.Column(db.String)

class Startup(db.Model):
    __tablename__ = "startups"
    startup_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    startup_name = db.Column(db.String)
    domain = db.Column(db.String)
    registration_type = db.Column(db.String)
    stage = db.Column(db.String)

# -------------------
# Routes
# -------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_document():
    user_id = request.form.get("user_id")
    doc_type = request.form.get("doc_type")

    if not user_id or not doc_type:
        return jsonify({"error": "Missing user_id or doc_type"}), 400

    user = User.query.get(user_id)
    startup = Startup.query.filter_by(user_id=user_id).first()
    if not user or not startup:
        return jsonify({"error": "User or startup not found"}), 404

    # Optional additional fields from form
    other_party = request.form.get("other_party")
    partner_name = request.form.get("partner_name")
    authority = request.form.get("authority")
    subject = request.form.get("subject")
    purpose = request.form.get("purpose")

    # Generate document
    if doc_type == "nda":
        file_path = generate_nda(user, startup, other_party, purpose)
    elif doc_type == "pitch_deck":
        file_path = generate_pitch_deck(user, startup)
    elif doc_type == "mou":
        file_path = generate_mou(user, startup, partner_name, purpose)
    elif doc_type == "rti":
        file_path = generate_rti(user, startup, authority, subject, purpose)
    else:
        return jsonify({"error": "Invalid document type"}), 400

    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
