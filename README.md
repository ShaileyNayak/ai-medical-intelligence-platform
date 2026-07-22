# AI Medical Intelligence Platform

**Chest X-Ray Disease Classification • Grad-CAM Explainability • LLM Assistive Reporting • FastAPI • PostgreSQL • Docker**

> Educational / evaluation prototype for SN Matrix Software Pvt. Ltd. — **not** a certified medical device.

## Live demo

Add your deployment URL here after hosting (Render / Railway / HF Spaces / EC2).

## Features

- ResNet18 transfer-learning classifier (**Normal** vs **Pneumonia**)
- Grad-CAM heatmap overlays (pytorch-grad-cam + manual fallback)
- LLM assistive reports (OpenAI API or safe template fallback)
- REST API with OpenAPI docs at `/docs`
- Prediction history in SQLite (local) / PostgreSQL (Docker)
- React + Vite + Tailwind UI and optional Streamlit UI
- Docker Compose orchestration (db + backend + frontend)

## Repository layout

```
backend/              FastAPI service
frontend/             React dashboard
frontend_streamlit/   Streamlit alternative
model/                Training, Grad-CAM, checkpoints
data/                 Dataset (gitignored) + samples
docs/                 Architecture & project report
```

## Quick start (Docker — recommended)

```bash
cp .env.example .env
# Place or generate weights:
#   cd model && python bootstrap_checkpoint.py
docker compose up --build
```

- UI: http://localhost:3000  
- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

## Local development

### 1. Model checkpoint

```bash
cd model
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python prepare_demo_data.py
python train.py --data-dir ../data --epochs 3 --batch-size 8
# or quick bootstrap without training:
python bootstrap_checkpoint.py
```

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend (React)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 (Vite proxies `/api` and `/static` to the backend).

### 4. Streamlit (optional)

```bash
cd frontend_streamlit
pip install -r requirements.txt
set API_BASE=http://localhost:8000
streamlit run app.py
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness + model status |
| POST | `/api/predict` | Upload X-ray → prediction, heatmap, report |
| GET | `/api/history` | Paginated history |
| GET | `/api/history/{id}` | Prediction detail |
| DELETE | `/api/history/{id}` | Delete record |
| GET | `/api/report/{id}` | Fetch / regenerate LLM report |

## LLM configuration

Set in `.env`:

```
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.3
```

If `LLM_API_KEY` is empty, the API returns a structured template report with the medical disclaimer.

## Tests

```bash
cd backend
pytest -q
```

## Disclaimer

This platform is a technical demonstration built for educational and evaluation purposes. It is not a certified medical device and must not be used for clinical diagnosis or treatment decisions.

## License

MIT — see [LICENSE](LICENSE).
