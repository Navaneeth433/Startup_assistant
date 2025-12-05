from flask import Flask, render_template, request, jsonify
from matcher import load_schemes, match_schemes

app = Flask(__name__)

# Load schemes once at startup
schemes = load_schemes("startup_schemes_final.json")

@app.route("/")
def index():
    # Just render template — dropdowns are hardcoded in HTML
    return render_template("index.html")

@app.route("/match", methods=["POST"])
def match():
    data = request.json
    domain = data.get("domain")
    registration = data.get("registration")
    stage = data.get("stage")

    results = match_schemes(schemes, domain, registration, stage)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
