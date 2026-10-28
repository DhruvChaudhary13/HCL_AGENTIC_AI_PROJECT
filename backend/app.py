"""Flask app: serves the frontend and exposes a small JSON API."""
import os

from flask import Flask, jsonify, request, send_from_directory

import predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

app = Flask(__name__, static_folder=None)


@app.errorhandler(predictor.ModelNotReady)
def model_not_ready(exc):
    return jsonify(error=str(exc), code="model_not_ready"), 503


@app.get("/api/health")
def health():
    try:
        predictor.load_bundle()
        return jsonify(status="ok", model="loaded")
    except predictor.ModelNotReady as exc:
        # The server itself is fine; report that the model is missing.
        return jsonify(status="ok", model="missing", detail=str(exc))


@app.get("/api/schema")
def schema():
    return jsonify(predictor.get_schema())


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Request body must be a JSON object."), 400
    row, notes, error = predictor.validate(data)
    if error:
        return jsonify(error=error), 400
    return jsonify(predictor.predict(row, notes))


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
