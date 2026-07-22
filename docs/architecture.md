# Architecture Diagram

```text
      +-----------------------+
      |        CLIENT         |
      |  React / Streamlit UI |
      |  Upload X-ray, view    |
      |  prediction & report   |
      +-----------+-----------+
                  |  HTTPS (REST / JSON)
                  v
      +-----------------------+
      |     API GATEWAY       |
      |  FastAPI (Uvicorn)    |
      |  /api/predict         |
      |  /api/history /health |
      +------+---------+-----+
             |         |
    +--------+         +--------+
    v                          v
+----------------+   +------------------------+
| INFERENCE       |   |  PERSISTENCE LAYER      |
| SERVICE         |   |  PostgreSQL / SQLite    |
| - CNN model     |   |  via SQLAlchemy ORM     |
| - Preprocessing |   |  predictions / reports  |
| - Grad-CAM XAI  |   +------------------------+
+--------+--------+
         | prediction + heatmap + confidence
         v
+------------------------------+
|    LLM REPORTING SERVICE      |
|  Structured prompt → OpenAI   |
|  or template fallback         |
+------------------------------+

Cross-cutting: Docker Compose, structured logging,
config via .env, CORS-secured API, static/media volume.
```
