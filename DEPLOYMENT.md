# How to Deploy This Project

The project has two deployable pieces: the **Streamlit dashboard** (`app/app.py`, serves
the fitted model) and the **pipeline itself** (`src/`, produces the artifacts). The
fitted model ships in `artifacts/intrusion_detection_model.joblib` (~2 MB) — no
retraining is needed to deploy; the UNSW-NB15 data files are NOT needed at runtime.

## 1. Local deployment (fastest)

```bash
cd Cybersecurity_Intrusion_Detection
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app/app.py          # http://localhost:8501
```
The dashboard opens at http://localhost:8501 with 4 pages: Overview, EDA,
Model performance, Live prediction (upload a CSV of UNSW-NB15-shaped flow rows and
get attack probabilities + alerts at the fixed threshold 0.480).

## 2. Streamlit Community Cloud (free public URL)

1. Put the project on GitHub (private repo works too):
   ```bash
   git init
   git add .
   git commit -m "Intrusion detection project"
   git remote add origin https://github.com/YOUR-NAME/EDR.git
   git push -u origin main
   ```
   - Exclude `data/` before committing (the downloader refetches it):
     add `data/` to `.gitignore`.
   - `artifacts/multiclass_attack_classifier.joblib` is ~52 MB — under GitHub's
     100 MB hard limit, but push it with Git LFS if the push fails.
2. Go to https://share.streamlit.io, sign in with GitHub → **New app**.
3. Choose the repo, branch `main`, main file `app/app.py` → **Deploy**.
4. Cloud runs `pip install -r requirements.txt` automatically. The app makes no
   network calls, so no secrets configuration is needed.

## 3. Docker deployment (any host: Render, Fly.io, your own server)

`Dockerfile`:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
```bash
docker build -t intrusion-detection .
docker run -p 8501:8501 intrusion-detection
```
On Render: new Web Service → deploy from the GitHub repo → port 8501. On a VPS,
put nginx in front with a reverse proxy to port 8501 and HTTPS.

## 4. Headless inference (batch / API)

No dashboard needed:
```bash
python -m src.predict --demo                  # 10-record built-in demo
python -m src.predict --batch my_unsw_rows.csv  # writes scored CSV
```
To wrap it as an HTTP API in production, put `IntrusionPredictor.load()` (from
`src/predict.py`, threshold 0.480 baked in via metadata) behind FastAPI — the model
object accepts raw UNSW-NB15-shaped rows; keep `id/label/attack_cat` out of the
inputs.

## 5. What to remember

- The model is benchmark-grade, not production-grade: FPR 27.47% on UNSW-NB15's
  test shift (honest limitation) — calibrate the threshold per deployment
  environment before any real use.
- Paper trail for examiners: notebook, report (DOCX+PDF), slides, ZIP.
- Fresh full reproduction: `python -m src.download_data && python -m src.train`
  (~6.2 min, random_state=42 → identical numbers).
