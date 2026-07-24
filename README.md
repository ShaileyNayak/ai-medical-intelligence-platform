# AI Medical Intelligence Platform

Multi-disease medical imaging demo: upload a scan, run a CNN, get a **Grad-CAM** heatmap and an assistive text report, then browse history in a React dashboard.

**Not a medical device.** Educational / portfolio prototype only — not for clinical diagnosis or treatment.

| | |
|---|---|
| Backend | Python 3.11 · FastAPI · PyTorch · SQLAlchemy |
| Frontend | React 18 · Vite · Tailwind CSS |
| Database | SQLite (local) · PostgreSQL (Docker / production) |
| Deploy | Docker Compose · Render Blueprint (`render.yaml`) |
| License | [MIT](LICENSE) |

---

## Features

- **Scan types:** Chest X-ray · Brain MRI · Skin lesion (stretch)
- **Inference + Grad-CAM** per module via a model registry
- **LLM or template reports** (OpenAI / Gemini when configured; stub template otherwise)
- **React UI:** Overview dashboard, New Analysis, History (by category), Analytics
- **REST API** with auto docs at `/docs`
- **Docker Compose** for API + UI + Postgres

### Module status

| Module | `scan_type` | Labels | Weights in repo |
|--------|-------------|--------|-----------------|
| Chest X-ray | `chest_xray` | Normal, Pneumonia *(binary checkpoint)*; architecture also supports COVID-19, Tuberculosis | ✅ `backend/model_weights/chest_xray/best_model.pth` |
| Brain MRI | `brain_mri` | Tumor, No Tumor | ❌ Train and place `best_model.pth` |
| Skin lesion | `skin_lesion` | Malignant, Benign | ❌ Optional / stretch |

---

## Quick start (local)

### Prerequisites

- **Python 3.11** (recommended; 3.12+ may fail on some wheels)
- **Node.js 18+** and npm
- Git

### 1. Backend

```bash
cd backend
py -3.11 -m venv venv          # Windows: prefer py -3.11
# Windows:  venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
copy .env.example .env         # Windows; or: cp .env.example .env

uvicorn app.main:app --reload --port 8000
```

- API: http://127.0.0.1:8000  
- Swagger: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/api/health → `{"status":"ok", ...}`

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 → **New Analysis** → choose scan type → upload an image → submit.

If the browser shows a CORS error, ensure `CORS_ORIGINS` in `backend/.env` includes `http://localhost:5173`.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness + per-module load status |
| `POST` | `/api/predict` | Multipart: `file` + `scan_type` → predictions, heatmap URL, report |
| `GET` | `/api/history` | Paginated history; optional `scan_type` |
| `GET` | `/api/history/summary` | Counts / conditions / avg confidence by category |
| `GET` | `/api/history/{id}/report` | Full report + media URLs for one prediction |

Interactive docs: `/docs`.

---

## Repository layout

```text
ai-medical-intelligence-platform/
├── backend/                 # FastAPI app, models, tests, Dockerfile
│   ├── app/
│   │   ├── api/             # predict, history, health routes
│   │   ├── models/          # chest_xray, brain_mri, skin_lesion + registry
│   │   ├── services/        # inference, Grad-CAM, LLM
│   │   ├── db/              # SQLAlchemy + schemas
│   │   └── llm/             # report generator
│   ├── model_weights/       # <scan_type>/best_model.pth
│   ├── static/              # uploads + heatmaps
│   └── tests/
├── frontend/                # React (Vite) dashboard
├── notebooks/               # Colab-friendly training notebooks
├── docs/                    # Architecture notes, screenshots
├── docker-compose.yml
├── render.yaml
├── PROJECT_SPEC.md          # Architecture source of truth
└── README.md
```

---

## Docker

With Docker Desktop running, from the project root:

```bash
# Stop local uvicorn / Vite first if ports 8000 / 3000 are in use
docker compose up --build
```

| Service | URL |
|---------|-----|
| Backend | http://localhost:8000/docs |
| Frontend | http://localhost:3000 |
| Postgres | `localhost:5432` (user/db from `.env.example`) |

Copy root `.env.example` → `.env` before the first Compose run if needed.

---

## Training models

Training needs a GPU (Google Colab is fine). Notebooks:

- `notebooks/train_chest_xray.ipynb`
- `notebooks/train_brain_mri.ipynb`

**Typical flow**

1. Download a public dataset (e.g. Kaggle *Chest X-Ray Images (Pneumonia)*, brain tumor MRI).
2. Upload the notebook to Colab → Runtime → GPU.
3. Train → download `best_model.pth`.
4. Place weights:

```text
backend/model_weights/chest_xray/best_model.pth
backend/model_weights/brain_mri/best_model.pth
backend/model_weights/skin_lesion/best_model.pth   # optional
```

If a `.pth` is **> 100 MB**, use [Git LFS](https://git-lfs.com/):

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes backend/model_weights/
```

### Model metrics (chest X-ray)

**Chest X-ray:** Accuracy 100%, Precision 100%, Recall 100%, AUC 1.0 (evaluated on a held-out test set of **n=80**; train/val/test are content-disjoint. These numbers come from a synthetic demo dataset and should be interpreted cautiously — validate on a larger real-world test set, e.g. Kaggle Chest X-Ray Images (Pneumonia), before drawing performance conclusions).

| Metric | Value |
|--------|-------|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| AUC | 1.0 |
| Test set size | 80 |

Source: `backend/model_weights/chest_xray/metrics_report.json`.

---

## Tests

```bash
cd backend
# venv active
pytest
```

Expected: all tests pass (currently **36**).

---

## Environment variables

See `backend/.env.example` and root `.env.example`.

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | SQLite or Postgres connection string |
| `CHEST_XRAY_MODEL_PATH` / `BRAIN_MRI_MODEL_PATH` / `SKIN_LESION_MODEL_PATH` | Paths to `.pth` files |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins |
| `LLM_PROVIDER` | `stub` (default), or OpenAI / Gemini when keyed |
| `LLM_API_KEY` | Required for live LLM reports |
| `VITE_API_BASE_URL` | Frontend build-time API origin (Docker / Vercel / Render) |

---

## Deploy

### Backend + DB (Render)

1. Connect this repo on [Render](https://render.com) (Blueprint via `render.yaml`, or Web Service with root `backend/`).
2. Attach a PostgreSQL instance; set `DATABASE_URL`.
3. Set `CORS_ORIGINS` to your frontend origin and optional `LLM_API_KEY`.
4. Confirm https://\<your-backend\>/docs and `/api/health`.

### Frontend (Vercel or Render static)

1. Root directory: `frontend/`
2. Build: `npm install && npm run build`
3. Env: `VITE_API_BASE_URL=https://\<your-backend\>` (no trailing slash)
4. Redeploy after changing that variable (it is baked in at build time).

After deploy, replace the placeholders below:

| | URL |
|---|-----|
| Backend | _add Render URL_ |
| Frontend | _add Vercel / Render URL_ |

---

## Screenshots

Add captures under `docs/screenshots/` (upload page, result + heatmap, history, overview) and link them here after you record a live demo.

---

## Disclaimer

This project is for **learning and portfolio demonstration**. Model outputs and generated reports are assistive only and must be confirmed by a qualified clinician. Do not use for real patient care.

---

## License

[MIT](LICENSE) © 2026 Shailey Nayak
