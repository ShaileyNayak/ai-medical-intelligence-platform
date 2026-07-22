# PROJECT_SPEC.md — AI Medical Intelligence Platform

Concise source of truth for architecture and scaffolding. Reference this file in future prompts.

---

## 1. Multi-disease scope

Three **independent** imaging modules. Each has its own trained model file, its own Grad-CAM implementation (target layer + overlay), and is selected via a `scan_type` field sent from the frontend. The backend routes inference to the matching module.

| Module | `scan_type` | Task | Labels |
|--------|-------------|------|--------|
| **1. Chest X-ray** | `chest_xray` | Multi-label classifier | `Normal`, `Pneumonia`, `COVID-19`, `Tuberculosis` |
| **2. Brain MRI** | `brain_mri` | Binary classifier | `Tumor`, `No Tumor` |
| **3. Skin lesion** *(optional / stretch)* | `skin_lesion` | Binary classifier | `Malignant`, `Benign` |

**Pipeline (every module):** upload image + `scan_type` → module CNN inference → module Grad-CAM → LLM assistive report → persist history.

**Not a medical device.** Educational / portfolio prototype only; every report must include a clinical disclaimer.

---

## 2. Tech stack

| Layer | Stack |
|-------|--------|
| Backend | Python 3.11, **FastAPI**, Uvicorn, **PyTorch** / torchvision, **SQLAlchemy** 2.x, Alembic, Pydantic |
| Frontend | **React 18**, **Vite**, **Tailwind CSS**, Axios, React Router |
| Database | **PostgreSQL** (Docker / Render), **SQLite** (local) |
| XAI / LLM | Grad-CAM **per module**; OpenAI or Gemini report generator (+ stub template) |
| Deploy | **Docker** / Docker Compose, Render Blueprint (`render.yaml`) |

---

## 3. Folder structure

### Root

```
ai-medical-intelligence-platform/
├── backend/
├── frontend/
├── models/                     # One subfolder per disease module (training + assets)
│   ├── chest_xray/
│   ├── brain_mri/
│   └── skin_lesion/            # Stretch
├── notebooks/
├── data/                       # Datasets (gitignored) + samples/ (per modality)
│   ├── chest_xray/
│   ├── brain_mri/
│   └── skin_lesion/
├── docs/
├── docker-compose.yml
├── render.yaml
├── .env.example
├── PROJECT_SPEC.md
├── README.md
└── LICENSE
```

### Shared / training modules (`models/`)

One subfolder per module. Each owns training scripts, label maps, and docs for that modality.

```
models/
├── chest_xray/
│   ├── train.py                # Multi-label Normal / Pneumonia / COVID-19 / Tuberculosis
│   ├── labels.py               # LABEL_NAMES, thresholds
│   └── README.md
├── brain_mri/
│   ├── train.py                # Binary Tumor / No Tumor
│   ├── labels.py
│   └── README.md
└── skin_lesion/                # Optional / stretch
    ├── train.py                # Binary Malignant / Benign
    ├── labels.py
    └── README.md
```

### Backend (`backend/`)

Inference packages mirror `models/`: one package per module (own weights path + Grad-CAM).

```
backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_predict.py      # POST /api/predict  (requires scan_type + file)
│   │   ├── routes_history.py      # GET /api/history   (optional scan_type filter)
│   │   └── routes_health.py       # GET /api/health    (per-module load status)
│   ├── core/
│   │   ├── config.py              # Paths to each module's .pth
│   │   └── logging.py
│   ├── db/
│   │   ├── database.py
│   │   ├── models.py              # ORM Prediction (+ scan_type)
│   │   ├── crud.py
│   │   └── schemas.py             # Pydantic I/O
│   ├── llm/
│   │   └── report_generator.py    # Prompt conditioned on scan_type + labels
│   ├── models/                    # Runtime ML packages — one subfolder per module
│   │   ├── registry.py            # get_model(scan_type)
│   │   ├── chest_xray/
│   │   │   ├── __init__.py
│   │   │   ├── model.py           # Load + inference
│   │   │   └── gradcam.py
│   │   ├── brain_mri/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── gradcam.py
│   │   └── skin_lesion/           # Stretch
│   │       ├── __init__.py
│   │       ├── model.py
│   │       └── gradcam.py
│   ├── services/
│   │   ├── inference_service.py   # Dispatch by scan_type → get_model()
│   │   ├── gradcam_service.py     # Dispatch by scan_type → module Grad-CAM
│   │   └── llm_service.py
│   └── utils/
│       ├── image_utils.py
│       └── validators.py          # Allowed scan_type values
├── model_weights/                 # Deployed .pth files — one subfolder per module
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
│   │   └── client.js              # predict({ scanType, file }), history, health
│   ├── components/
│   │   ├── UploadCard.jsx         # Drag/drop + scan_type dropdown (required)
│   │   ├── PredictionResult.jsx   # Multi-label (chest) or binary display
│   │   ├── HeatmapViewer.jsx
│   │   ├── ReportPanel.jsx
│   │   └── HistoryTable.jsx       # Optional scan_type filter
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

## 4. Module contract

| Concern | Rule |
|---------|------|
| Model file | One `.pth` (or equivalent) under `backend/model_weights/<module>/` |
| Grad-CAM | Implemented inside `backend/app/models/<module>/gradcam.py` with that CNN’s target layer |
| Selection | Frontend sends `scan_type`; backend must not guess modality from pixels alone |
| Isolation | Modules do not share weights or Grad-CAM layers; shared code lives in `services/` / `utils/` only |

---

## 5. API contract

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
- Errors: `400` bad file / unknown `scan_type` · `422` validation · `500` pipeline failure · `501` if stretch module not enabled.

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

## 6. Database schema — `predictions`

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

## 7. Scan-type constants (reference)

```text
chest_xray  → labels: Normal, Pneumonia, COVID-19, Tuberculosis  (multi-label)
brain_mri   → labels: Tumor, No Tumor                            (binary)
skin_lesion → labels: Malignant, Benign                          (binary, stretch)
```

Frontend: required dropdown / selector on upload.  
Backend: `validators` reject unknown values; `inference_service` / `gradcam_service` dispatch to `app.models.<scan_type>`.

---

## 8. Non-goals (v1)

- DICOM / PACS integration  
- Regulatory (FDA/CE) validation  
- User auth (optional later)  
- Shipping skin-lesion weights as required for MVP (stretch only)

---

*Update this file when the API or schema changes so agents and contributors stay aligned.*
