# Separate Render service notes (manual setup)

Use these if you prefer creating services in the dashboard instead of the root `render.yaml` Blueprint.

## Backend (Docker web service)

| Setting | Value |
|---------|--------|
| Runtime | Docker |
| Dockerfile path | `./backend/Dockerfile` |
| Docker build context directory | `./backend` |
| Health check path | `/api/health` |
| Instance type | Free / Starter |

### Start command

Defined in the image (`CMD ["./start.sh"]`). Honors Render’s `PORT` env var.

### Environment

```text
DATABASE_URL=<from Render Postgres — Internal Database URL>
LLM_API_KEY=<secret — leave empty for stub reports>
LLM_PROVIDER=stub
LLM_MODEL=gpt-4o-mini
MODEL_PATH=/app/model_weights/best_model.pth
MODEL_URL=<optional HTTPS URL to best_model.pth>
UPLOAD_DIR=/app/static/uploads
HEATMAP_DIR=/app/static/heatmaps
CORS_ORIGINS=https://<your-frontend>.onrender.com
LOG_LEVEL=INFO
```

## Frontend (Static site)

| Setting | Value |
|---------|--------|
| Root directory | `frontend` |
| Build command | `npm install && npm run build` |
| Publish directory | `dist` |
| Rewrite rule | `/*` → `/index.html` |

### Environment (build-time)

```text
VITE_API_BASE_URL=https://<your-backend>.onrender.com
```

## Postgres

Create a Render PostgreSQL database and link `DATABASE_URL` to the backend service (or paste the connection string manually).
