"""FastAPI backend that serves the demo frontend and scores UNSW-NB15-shaped rows
with the fitted pipeline artifact. One command starts the whole demo:

    python -m app.api          # then open http://localhost:8000

Endpoints:
  GET  /                -> the HTML frontend (frontend/index.html)
  POST /api/predict     -> multipart CSV upload; returns per-row probabilities
  GET  /api/metadata    -> model metadata (threshold, params, versions)
"""
from pathlib import Path
import io
import json

import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
FRONTEND = ROOT / "frontend"

THRESHOLD = 0.480  # fixed validation-side; never changed after testing

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="Intrusion Detection Demo", version="1.0")


@app.get("/api/metadata")
def metadata():
    meta_path = ARTIFACTS / "model_metadata.json"
    if not meta_path.exists():
        raise HTTPException(503, "Artifact metadata missing - run python -m src.train first")
    return JSONResponse(json.loads(meta_path.read_text(encoding="utf-8")))


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    model_path = ARTIFACTS / "intrusion_detection_model.joblib"
    if not model_path.exists():
        raise HTTPException(503, "Model artifact missing - run python -m src.train first")
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not read CSV: {exc}")
    if df.empty:
        raise HTTPException(400, "Uploaded CSV is empty")
    if len(df) > 5000:
        df = df.head(5000)  # keep the free tier container happy
    drop_cols = [c for c in ("id", "label", "attack_cat") if c in df.columns]
    X = df.drop(columns=drop_cols)
    try:
        model = joblib.load(model_path)
        proba = model.predict_proba(X)[:, 1]
    except Exception as exc:
        raise HTTPException(422, f"The model could not score these rows: {exc}")
    rows = []
    for i, p in enumerate(proba):
        r = {}
        if "id" in df.columns:
            r["id"] = int(df.iloc[i]["id"])
        if "attack_cat" in df.columns:
            r["attack_cat"] = str(df.iloc[i]["attack_cat"])
        if "label" in df.columns:
            r["actual"] = int(df.iloc[i]["label"])
        r["attack_probability"] = round(float(p), 4)
        r["alert"] = int(p >= THRESHOLD)
        rows.append(r)
    alerts = sum(r["alert"] for r in rows)
    return {"threshold": THRESHOLD, "n_rows": len(rows), "alerts": alerts,
            "accuracy": round(sum(r["alert"] == r["actual"] for r in rows if "actual" in r)
                              / max(1, sum(1 for r in rows if "actual" in r)), 4),
            "predictions": rows}


@app.get("/")
def index():
    page = FRONTEND / "index.html"
    if not page.exists():
        raise HTTPException(404, "Frontend missing")
    return FileResponse(str(page))


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
