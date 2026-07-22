# Project Report — Advanced AI Medical Intelligence Platform

**Version:** 1.0  
**Date:** 22 July 2026  
**Prepared for:** AI/ML Engineer – Technical Evaluation, SN Matrix Software Pvt. Ltd.  
**Domain:** Medical Image Analysis (Chest X-Ray Disease Classification)

## 1. Overview

End-to-end platform that:

1. Classifies chest X-rays as **Normal** or **Pneumonia** with a ResNet18 CNN.
2. Explains predictions with **Grad-CAM** heatmaps.
3. Generates an assistive narrative via an **LLM** (or template fallback).
4. Exposes functionality through a **FastAPI** REST API.
5. Persists history in **PostgreSQL/SQLite**.
6. Deploys via **Docker Compose** with a React UI.

This is an educational demonstration, not a medical device. Every report includes a clinical disclaimer.

## 2. Architecture

```
Client (React / Streamlit)
        | HTTPS REST/JSON
FastAPI (Uvicorn)
   ├── Inference Service (ResNet18)
   ├── Grad-CAM XAI Service
   ├── LLM Reporting Service
   └── Persistence (SQLAlchemy → PostgreSQL/SQLite)
```

Cross-cutting: Docker Compose, `.env` config, CORS, structured logging with request IDs, static volumes for uploads/heatmaps.

## 3. Technology Stack

| Area | Technologies |
|------|----------------|
| Deep Learning | PyTorch, torchvision, ResNet18, Pillow, OpenCV, scikit-learn |
| XAI | pytorch-grad-cam, manual Grad-CAM hooks |
| LLM | OpenAI API (gpt-4o-mini), template fallback |
| API | FastAPI, Uvicorn, Pydantic, python-multipart |
| Database | PostgreSQL / SQLite, SQLAlchemy, Alembic |
| Frontend | React, Vite, Tailwind, Axios, Recharts; Streamlit alternative |
| DevOps | Docker, Docker Compose, nginx |
| Quality | pytest, Black, Ruff |

## 4. Database

`predictions`: id, image_filename, predicted_label, confidence_score, heatmap_path, llm_report, model_version, created_at, user_id (nullable).

Optional `users` table for future auth.

## 5. API Design

- `GET /health`
- `POST /api/predict`
- `GET /api/history`, `GET /api/history/{id}`, `DELETE /api/history/{id}`
- `GET /api/report/{id}?regenerate=`

Interactive docs: `/docs`, `/redoc`.

## 6. Grad-CAM & LLM

Grad-CAM uses the last ResNet18 convolutional block (`layer4[-1]`), overlays a jet colormap, and saves PNGs under `static/heatmaps/`.

LLM prompts inject label, confidence, and region description; temperature ≤ 0.3; disclaimer always appended.

## 7. How to Run

See root `README.md` for Docker and local instructions.

## 8. Deliverables

- Source: `backend/`, `frontend/`, `frontend_streamlit/`, `model/`
- Checkpoint: `model/checkpoints/best_model.pt`
- Docker: `Dockerfile`s + `docker-compose.yml`
- Docs: this report + architecture notes

## 9. Disclaimer

This report and the associated software are AI-assisted technical demonstrations for educational/evaluation purposes only. They are not medical diagnoses and must not guide clinical care without licensed clinician review.
