#!/bin/bash

# Installera ffmpeg om det saknas
echo "[INFO] Checking for ffmpeg..."
ffmpeg -version || echo "[INFO] ffmpeg not found, installing..."
apt-get update
apt-get install -y ffmpeg
echo "[INFO] Installed version:"
ffmpeg -version

# Om en del paket saknas, installera dem här
pip install uvicorn[standard]

# Starta appen
echo "[INFO] Starting JBGtranscription API via Gunicorn"

# Kör med 4 workers och uvicorn workers
exec gunicorn main:app \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 600
