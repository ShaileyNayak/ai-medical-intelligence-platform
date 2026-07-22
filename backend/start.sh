#!/bin/sh
set -eu

PORT="${PORT:-8000}"
MODEL_PATH="${MODEL_PATH:-/app/model_weights/best_model.pth}"

mkdir -p /app/static/uploads /app/static/heatmaps /app/model_weights

# Optional: fetch checkpoint if missing (set MODEL_URL to a direct HTTPS link)
if [ ! -f "$MODEL_PATH" ] && [ -n "${MODEL_URL:-}" ]; then
  echo "Downloading model weights from MODEL_URL → $MODEL_PATH"
  curl -fsSL "$MODEL_URL" -o "$MODEL_PATH"
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "WARNING: No weights at $MODEL_PATH — API will start with an unloaded/random head."
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
