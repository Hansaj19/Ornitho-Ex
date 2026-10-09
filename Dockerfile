# ── Ornitho-Ex · Streamlit Docker Image ─────────────────────────────────────
# Build:  docker build -t ornitho-ex .
# Run:    docker run -p 8501:8501 ornitho-ex
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# System deps for librosa / soundfile / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Streamlit listens on 8501 by default
EXPOSE 8501

# Health-check so container orchestrators know when the app is up
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
