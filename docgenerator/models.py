from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# -----------------------
# User Model
# -----------------------
class User(db.Model):
    __tablename__ = "users"   # Your table name in Postgres
    user_id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String, nullable=False)
    email = db.Column(db.String, unique=True, nullable=False)

# -----------------------
# Startup Model
# -----------------------
class Startup(db.Model):
    __tablename__ = "startups"   # Your table name in Postgres
    startup_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=False)
    startup_name = db.Column(db.String, nullable=False)
    domain = db.Column(db.String)
    registration_type = db.Column(db.String)
    stage = db.Column(db.String)
    funding_amount = db.Column(db.Float)
    team_size = db.Column(db.Integer)
    location = db.Column(db.String)
    website = db.Column(db.String)
    problem_statement = db.Column(db.String)
    vision = db.Column(db.String)
