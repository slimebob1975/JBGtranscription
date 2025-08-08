#!/bin/bash

# Installera ffmpeg om det saknas
echo "[INFO] Checking for ffmpeg..."
ffmpeg -version || echo "[INFO] ffmpeg not found, installing..."
apt-get update
apt-get install -y ffmpeg libsndfile1
echo "[INFO] Installed version:"
ffmpeg -version

# Installera Whisper-modellen i fallback om inte cache finns (valfritt)
echo "[INFO] Starting JBGtranscription API via Gunicorn"

# Kör med 4 workers och uvicorn workers
exec gunicorn main:app \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 600
