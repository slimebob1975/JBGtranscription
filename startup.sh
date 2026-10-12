#!/bin/bash

# Installera ffmpeg om det saknas
echo "[INFO] Checking for ffmpeg..."
ffmpeg -version || echo "[INFO] ffmpeg not found, installing..."
apt-get update
apt-get install -y ffmpeg
echo "[INFO] Installed version:"
ffmpeg -version

# Starta appen
echo "[INFO] Starting JBGtranscription API via Gunicorn"

# En worker, inte flera. Jobbstatus hålls i processens eget minne medan
# resultatfilen ligger på delad disk, så med flera workers kan uppladdning,
# statusfrågor och nedladdning hamna i olika processer. Varje worker laddar
# dessutom en egen Whisper-modell, och arbetet är ändå CPU-bundet, så en extra
# worker kostar minne utan att ge något.
#
# Tidsgränsen måste rymma den längsta transkriberingen. Gunicorn startar om en
# worker som inte hört av sig inom --timeout, och en förfrågan som kommer in
# till en worker som just startas om besvaras med 502 av Azures gateway. En
# intervju på en timme tog 19 minuter på "medel" och 74 minuter på "mycket
# hög", vilket är långt mer än de 601 sekunder som gällde tidigare.
exec gunicorn main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 7200 \
    --graceful-timeout 120
