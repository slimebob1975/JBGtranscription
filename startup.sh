#!/bin/bash

# Installera Whisper-modellen i fallback om inte cache finns (valfritt)
echo "[INFO] Starting JBGtranscription API via Gunicorn"

# Kör med 4 workers och uvicorn workers
exec gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 600
