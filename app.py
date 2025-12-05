import os
import logging
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError
from models import db, User, Startup

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- App Initialization ---
app = Flask(__name__)
CORS(app, supports_credentials=True)   # ✅ allow cookies/sessions

# --- Configuration ---
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "a-strong-default-secret-key-for-dev")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:300234@localhost:5432/startup_assistant"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# Create database tables if they don't exist
with app.app_context():
    db.create_all()


# --- API Routes ---
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    required_fields = ["full_name", "email", "password", "startup_name", "domain", "registration_type", "stage"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        if User.query.filter_by(email=data["email"]).first():
            return jsonify({"error": "Email already registered"}), 409

        new_user = User(
            full_name=data["full_name"],
            email=data["email"].lower(),
            password_hash=generate_password_hash(data["password"])
        )
        db.session.add(new_user)
        db.session.flush()

        new_startup = Startup(
            user_id=new_user.user_id,
            startup_name=data["startup_name"],
            domain=data["domain"],
            registration_type=data["registration_type"],
            stage=data["stage"],
            funding_amount=data.get("funding_amount"),
            team_size=data.get("team_size"),
            location=data.get("location"),
            website=data.get("website"),
            problem_statement=data.get("problem_statement"),
            vision=data.get("vision")
        )
        db.session.add(new_startup)
        db.session.commit()

        return jsonify({"message": "Signup successful"}), 201

    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "A database integrity error occurred."}), 500
    except Exception as e:
        db.session.rollback()
        logging.error(f"An unexpected error occurred during signup: {e}")
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or not data.get("email") or not data.get("password"):
        return jsonify({"error": "Email and password are required"}), 400

    user = User.query.filter_by(email=data["email"].lower()).first()

    if user and check_password_hash(user.password_hash, data["password"]):
        session["user_id"] = user.user_id  # ✅ store session
        return jsonify({
            "message": "Login successful", 
            "user_id": user.user_id
        }), 200
    
    return jsonify({"error": "Invalid email or password"}), 401


@app.route("/user/<int:user_id>", methods=["GET"])
def get_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    startup = Startup.query.filter_by(user_id=user_id).first()

    return jsonify({
        "user_id": user.user_id,
        "full_name": user.full_name,
        "email": user.email,
        "startup": {
            "startup_name": startup.startup_name if startup else None,
            "domain": startup.domain if startup else None,
            "registration_type": startup.registration_type if startup else None,
            "stage": startup.stage if startup else None,
            "funding_amount": startup.funding_amount if startup else None,
            "team_size": startup.team_size if startup else None,
            "location": startup.location if startup else None,
            "website": startup.website if startup else None,
            "problem_statement": startup.problem_statement if startup else None,
            "vision": startup.vision if startup else None
        }
    }), 200
# Check current session
@app.route("/current_user", methods=["GET"])
def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"logged_in": False}), 200
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({"logged_in": False}), 200
    
    return jsonify({
        "logged_in": True,
        "user_id": user.user_id,
        "full_name": user.full_name,
        "email": user.email
    }), 200



# ✅ NEW LOGOUT ROUTE
@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logout successful"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
