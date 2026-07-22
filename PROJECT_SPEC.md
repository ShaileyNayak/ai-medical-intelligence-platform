# PROJECT_SPEC.md — AI Medical Intelligence Platform

Concise source of truth for architecture and scaffolding. Reference this file in future prompts.

---

## 1. Project goal

Build a **multi-disease medical image analysis platform** with three imaging modules:

| Module | Modality | Task | Labels |
|--------|----------|------|--------|
| **Chest X-ray** | CXR | Multi-label classification | `Normal`, `Pneumonia`, `COVID-19`, `Tuberculosis` |
| **Brain MRI** | MRI | Binary classification | `Tumor`, `No Tumor` |
| **Skin lesion** *(stretch)* | Dermatoscopy | Binary classification | `Malignant`, `Benign` |

Each module: upload image → CNN inference → Grad-CAM (where applicable) → LLM assistive report → persist history.

**Not a medical device.** Educational / portfolio prototype only; every report must include a clinical disclaimer.

---

## 2. Tech stack

| Layer | Stack |
|-------|--------|
| Backend | Python 3.11, **FastAPI**, Uvicorn, **PyTorch** / torchvision, **SQLAlchemy** 2.x, Alembic, Pydantic |
| Frontend | **React 18**, **Vite**, **Tailwind CSS**, Axios, React Router |
| Database | **PostgreSQL** (Docker / Render), **SQLite** (local) |
| XAI / LLM | Grad-CAM per module; OpenAI or Gemini report generator (+ stub template) |
| Deploy | **Docker** / Docker Compose, Render Blueprint (`render.yaml`) |

---

## 3. Folder structure

### Root

```
ai-medical-intelligence-platform/
├── backend/
├── frontend/
├── model/                      # Shared training utilities / notebooks (optional)
├── notebooks/
├── data/                       # Datasets (gitignored) + samples/
├── docs/
├── docker-compose.yml
├── render.yaml
├── .env.example
├── PROJECT_SPEC.md
├── README.md
└── LICENSE
```

### Backend (`backend/`)

```
backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_predict.py      # POST /api/predict
│   │   ├── routes_history.py      # GET /api/history
│   │   └── routes_health.py       # GET /api/health
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── db/
│   │   ├── database.py
│   │   ├── models.py              # ORM Prediction
│   │   ├── crud.py
│   │   └── schemas.py             # Pydantic I/O
│   ├── llm/
│   │   └── report_generator.py
│   ├── models/                    # ML packages (per module)
│   │   ├── chest_xray/
│   │   │   ├── __init__.py
│   │   │   ├── ml_model.py        # Multi-label CNN (e.g. DenseNet/ResNet)
│   │   │   ├── gradcam.py
│   │   │   └── preprocess.py
│   │   ├── brain_mri/
│   │   │   ├── __init__.py
│   │   │   ├── ml_model.py        # Binary Tumor / No Tumor
│   │   │   ├── gradcam.py
│   │   │   └── preprocess.py
│   │   └── skin_lesion/           # Stretch goal
│   │       ├── __init__.py
│   │       ├── ml_model.py        # Binary Malignant / Benign
│   │       ├── gradcam.py
│   │       └── preprocess.py
│   ├── services/
│   │   ├── inference_service.py   # Dispatches by scan_type
│   │   ├── gradcam_service.py
│   │   └── llm_service.py
│   └── utils/
│       ├── image_utils.py
│       └── validators.py
├── model_weights/
│   ├── chest_xray/
│   │   └── best_model.pth
│   ├── brain_mri/
│   │   └── best_model.pth
│   └── skin_lesion/
│       └── best_model.pth
├── static/
│   ├── uploads/
│   └── heatmaps/
├── alembic/
├── tests/
├── Dockerfile
├── start.sh
├── requirements.txt
└── .env.example
```

### Frontend (`frontend/`)

```
frontend/
├── public/
│   └── favicon.ico
├── src/
│   ├── main.jsx
│   ├── App.jsx
│   ├── api/
│   │   └── client.js              # predict(scanType, file), history, health
│   ├── components/
│   │   ├── UploadCard.jsx         # Drag/drop + scan_type selector
│   │   ├── PredictionResult.jsx   # Single- or multi-label display
│   │   ├── HeatmapViewer.jsx
│   │   ├── ReportPanel.jsx
│   │   └── HistoryTable.jsx
│   ├── pages/
│   │   ├── Dashboard.jsx
│   │   └── History.jsx
│   ├── styles/
│   │   └── index.css
│   └── hooks/
│       └── usePrediction.js
├── package.json
├── vite.config.js
├── tailwind.config.js
├── nginx.conf
└── Dockerfile
```

---

## 4. API contract

Base path prefix: `/api`. Interactive docs: `/docs`.

### `GET /api/health`

**Response `200`**

```json
{
  "status": "ok",
  "models_loaded": {
    "chest_xray": true,
    "brain_mri": true,
    "skin_lesion": false
  },
  "version": "1.0.0"
}
```

### `POST /api/predict`

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `scan_type` | string | yes | `chest_xray` \| `brain_mri` \| `skin_lesion` |
| `file` | file | yes | Image (JPEG / PNG / WebP) |

**Behavior by `scan_type`**

| `scan_type` | Prediction shape |
|-------------|------------------|
| `chest_xray` | **Multi-label**: list of `{ label, confidence }` (sigmoid / thresholded) |
| `brain_mri` | **Single-label**: one `{ label, confidence }` (`Tumor` or `No Tumor`) |
| `skin_lesion` | **Single-label**: one `{ label, confidence }` (`Malignant` or `Benign`) |

**Response `200`**

```json
{
  "id": 12,
  "scan_type": "chest_xray",
  "predictions": [
    { "label": "Pneumonia", "confidence": 0.91 },
    { "label": "COVID-19", "confidence": 0.42 }
  ],
  "prediction_label": "[{\"label\":\"Pneumonia\",\"confidence\":0.91},{\"label\":\"COVID-19\",\"confidence\":0.42}]",
  "primary_label": "Pneumonia",
  "confidence": 0.91,
  "heatmap_url": "/static/heatmaps/….png",
  "image_url": "/static/uploads/….png",
  "report_text": "…plain-language summary…\n\nDisclaimer: This is not a medical diagnosis…",
  "created_at": "2026-07-22T10:15:32Z"
}
```

Notes:

- `predictions` is always an **array** (length 1 for binary modules; 1–N for chest multi-label positives / top scores).
- `prediction_label` mirrors the JSON-serialized form stored in the DB.
- `primary_label` / `confidence` = highest-confidence entry (convenience for UI).
- Errors: `400` bad file / unknown `scan_type` · `422` validation · `500` pipeline failure.

### `GET /api/history`

**Query:** `page` (default 1), `page_size` (default 20), optional `scan_type` filter.

**Response `200`**

```json
{
  "items": [
    {
      "id": 12,
      "scan_type": "brain_mri",
      "prediction_label": "[{\"label\":\"Tumor\",\"confidence\":0.87}]",
      "predictions": [{ "label": "Tumor", "confidence": 0.87 }],
      "primary_label": "Tumor",
      "confidence": 0.87,
      "heatmap_url": "/static/heatmaps/….png",
      "image_url": "/static/uploads/….png",
      "report_text": "…",
      "created_at": "2026-07-22T10:15:32Z"
    }
  ],
  "total": 42,
  "page": 1,
  "page_size": 20
}
```

---

## 5. Database schema — `predictions`

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer, PK, autoincrement | Primary key |
| `scan_type` | String(32), indexed, not null | `chest_xray` \| `brain_mri` \| `skin_lesion` |
| `image_path` | String(512), not null | Stored upload path / filename |
| `heatmap_path` | String(512), not null | Grad-CAM overlay path / filename |
| `prediction_label` | Text, not null | **JSON-serialized** list of `{label, confidence}` (supports multi-label) |
| `confidence` | Float, not null | Primary / max confidence (0–1) for sorting & quick display |
| `report_text` | Text, not null | LLM or template assistive report |
| `created_at` | DateTime(tz), server default now, indexed | Insert timestamp |

**Example `prediction_label` values**

```text
[{"label":"Pneumonia","confidence":0.91},{"label":"Tuberculosis","confidence":0.55}]
[{"label":"No Tumor","confidence":0.93}]
[{"label":"Malignant","confidence":0.81}]
```

SQLite locally (`DATABASE_URL=sqlite:///./predictions.db`); PostgreSQL in Docker / Render.

---

## 6. Scan-type constants (reference)

```text
chest_xray  → labels: Normal, Pneumonia, COVID-19, Tuberculosis  (multi-label)
brain_mri   → labels: Tumor, No Tumor                            (binary)
skin_lesion → labels: Malignant, Benign                          (binary, stretch)
```

---

## 7. Non-goals (v1)

- DICOM / PACS integration  
- Regulatory (FDA/CE) validation  
- User auth (optional later)  
- Shipping skin-lesion weights as required for MVP (stretch only)

---

*Update this file when the API or schema changes so agents and contributors stay aligned.*
