#!/bin/bash

# --- CONFIG ---
MEM_PER_WORKER_MB=1500  # Approximate memory usage per worker
TIMEOUT=600

# --- DETECT SYSTEM RESOURCES ---

# Get available memory (in MB)
AVAILABLE_RAM_MB=$(free -m | awk '/^Mem:/ {print $2}')

# Get number of CPU cores
CPU_CORES=$(nproc)

# --- CALCULATE WORKERS ---

# Estimate max workers based on RAM
WORKERS_BY_RAM=$((AVAILABLE_RAM_MB / MEM_PER_WORKER_MB))
# Ensure at least 1
WORKERS_BY_RAM=$((WORKERS_BY_RAM > 0 ? WORKERS_BY_RAM : 1))

# Final worker count: min(RAM-based, CPU-based)
WORKERS=$((WORKERS_BY_RAM < CPU_CORES ? WORKERS_BY_RAM : CPU_CORES))

# --- LOGGING ---
echo "[INFO] Available RAM: ${AVAILABLE_RAM_MB} MB"
echo "[INFO] CPU cores: ${CPU_CORES}"
echo "[INFO] Starting with $WORKERS worker(s)"

# --- START APP ---
exec gunicorn main:app \
    --workers $WORKERS \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout $TIMEOUT
