import io
import os
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

from model import Captioner, EVAL_TF
from vocab import Vocabulary

CKPT = Path(os.environ.get("CKPT", "checkpoints/captioner.pt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

MODEL, VOCAB, LOAD_ERROR = None, None, None


def load_model():
    global MODEL, VOCAB, LOAD_ERROR
    if not CKPT.exists():
        LOAD_ERROR = f"No trained model found at {CKPT}. Run train.py first."
        return
    try:
        ck = torch.load(CKPT, map_location=DEVICE)
        VOCAB = Vocabulary.from_itos(ck["itos"])
        MODEL = Captioner(len(VOCAB), pretrained=False, **ck["config"])
        MODEL.load_state_dict(ck["model"])
        MODEL.to(DEVICE).eval()
    except Exception as exc:  # corrupt or mismatched checkpoint
        LOAD_ERROR = f"Could not load {CKPT}: {exc}"


load_model()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(ready=MODEL is not None, device=str(DEVICE), error=LOAD_ERROR)


@app.post("/api/caption")
def caption():
    if MODEL is None:
        return jsonify(error=LOAD_ERROR), 503
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="Choose an image to caption."), 400
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(file.read()))).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file isn't a readable image. Try a JPG, PNG or WebP."), 400
    try:
        beam = max(1, min(5, int(request.form.get("beam", 3))))
    except ValueError:
        beam = 3

    tensor = EVAL_TF(img).unsqueeze(0).to(DEVICE)
    results = MODEL.generate(tensor, VOCAB, beam=beam, n_best=min(3, beam))
    return jsonify(caption=results[0][0], alternatives=[c for c, _ in results[1:]])


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is larger than 10 MB. Try a smaller file."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
