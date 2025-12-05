from flask import Flask, request, jsonify
from flask_cors import CORS
from matcher import match_schemes

app = Flask(__name__)
CORS(app, resources={r"/match_schemes": {"origins": "*"}})  # allow all origins

@app.route("/match_schemes", methods=["POST"])
def api_match_schemes():
    startup_profile = request.json
    results = match_schemes(startup_profile)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
