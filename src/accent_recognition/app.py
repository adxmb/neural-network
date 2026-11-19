"""FastAPI service that sers a trained accent model
uvicorn accent_recognition.app:app --host 0.0.0.0 --port 8000
"""
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .features import extract_features
from .model import NeuralNetwork

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.npz")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
INDEX_PATH = Path(__file__).parent / "static" / "index.html"
state = {}

@asynccontextmanager
async def lifespan(app):
    state["net"], state["scaler"], state["classes"] = NeuralNetwork.load(MODEL_PATH)
    yield
    state.clear()

app = FastAPI(title="Accent Recognition", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    # Read on every request, so edits to the page show up after a refresh
    return INDEX_PATH.read_text(encoding="utf-8")

@app.get("/health")
def health():
    return {"status": "ok", "classes": state["classes"]}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 10 MB)")

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            features = extract_features(tmp.name)
        except Exception as exc:
            raise HTTPException(400, f"Could not process audio: {exc}")

    x = state["scaler"].transform(features.reshape(1, -1))
    probs = state["net"].predict_proba(x)[0]
    best = int(probs.argmax())
    return {
        "accent": state["classes"][best],
        "confidence": float(probs[best]),
        "probabilities": {c: float(p) for c, p in zip(state["classes"], probs)},
    }