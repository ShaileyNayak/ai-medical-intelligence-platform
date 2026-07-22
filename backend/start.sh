#!/bin/sh
set -eu

PORT="${PORT:-8000}"
MODEL_PATH="${MODEL_PATH:-/app/model_weights/chest_xray/best_model.pth}"
CHEST_XRAY_MODEL_PATH="${CHEST_XRAY_MODEL_PATH:-$MODEL_PATH}"

mkdir -p /app/static/uploads /app/static/heatmaps \
  /app/model_weights/chest_xray \
  /app/model_weights/brain_mri \
  /app/model_weights/skin_lesion

# Optional: fetch chest checkpoint if missing (set MODEL_URL to a direct HTTPS link)
if [ ! -f "$CHEST_XRAY_MODEL_PATH" ] && [ -n "${MODEL_URL:-}" ]; then
  echo "Downloading model weights from MODEL_URL → $CHEST_XRAY_MODEL_PATH"
  curl -fsSL "$MODEL_URL" -o "$CHEST_XRAY_MODEL_PATH"
fi

if [ ! -f "$CHEST_XRAY_MODEL_PATH" ]; then
  echo "WARNING: No chest_xray weights at $CHEST_XRAY_MODEL_PATH — API will start with an unloaded/random head."
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
