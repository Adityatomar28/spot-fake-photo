"""
demo_server.py — optional "impressive" extra.

Tiny local Flask server that:
  1. Serves a single HTML page using getUserMedia to access the camera.
  2. Receives a JPEG frame via POST, runs it through the same feature
     extraction + model used in predict.py, and returns the screen-probability.

Run:
    pip install flask
    python demo_server.py
Then open http://localhost:5000 in your browser.
"""

import base64
import os
import pickle

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory

from features import extract_features

FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "frontend", "dist"))

if os.path.exists(FRONTEND_DIST):
    app = Flask(__name__, static_folder=FRONTEND_DIST, static_url_path="")
else:
    app = Flask(__name__)

with open("model.pkl", "rb") as f:
    BUNDLE = pickle.load(f)
SCALER = BUNDLE["scaler"]
CLF = BUNDLE["clf"]
THRESHOLD = BUNDLE.get("threshold", 0.5)

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Spot the Fake Photo — Live Demo</title>
<style>
  body { font-family: -apple-system, sans-serif; background:#111; color:#eee;
         display:flex; flex-direction:column; align-items:center; padding:24px; }
  video, canvas { border-radius:8px; max-width:90vw; }
  #result { font-size:28px; margin-top:16px; font-weight:600; }
  .real { color:#4ade80; }
  .screen { color:#f87171; }
  button { margin-top:12px; padding:10px 20px; font-size:16px; cursor:pointer;
           border-radius:6px; border:none; background:#3b82f6; color:white; }
</style>
</head>
<body>
  <h2>Spot the Fake Photo — Live Demo</h2>
  <video id="video" autoplay playsinline width="480" height="360"></video>
  <button id="capture">Check this frame</button>
  <div id="result">—</div>
  <canvas id="canvas" width="480" height="360" style="display:none;"></canvas>

<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const resultEl = document.getElementById('result');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => { video.srcObject = stream; })
  .catch(err => { resultEl.textContent = 'Camera error: ' + err; });

document.getElementById('capture').onclick = async () => {
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

  resultEl.textContent = 'Checking...';
  resultEl.className = '';

  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataUrl })
  });
  const data = await res.json();

  if (data.prob > 0.5) {
    resultEl.textContent = `PHOTO OF A SCREEN (score ${data.prob.toFixed(3)})`;
    resultEl.className = 'screen';
  } else {
    resultEl.textContent = `REAL PHOTO (score ${data.prob.toFixed(3)})`;
    resultEl.className = 'real';
  }
};
</script>
</body>
</html>
"""


@app.route("/")
def index():
    if os.path.exists(FRONTEND_DIST):
        return send_from_directory(FRONTEND_DIST, "index.html")
    return Response(PAGE, mimetype="text/html")


@app.errorhandler(404)
def not_found(e):
    if os.path.exists(FRONTEND_DIST):
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({"error": "Not found"}), 404


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    data_url = payload["image"]
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    feats = extract_features(img).reshape(1, -1)
    feats_scaled = SCALER.transform(feats)
    prob = float(CLF.predict_proba(feats_scaled)[0, 1])

    return jsonify({"prob": prob, "threshold": THRESHOLD})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
