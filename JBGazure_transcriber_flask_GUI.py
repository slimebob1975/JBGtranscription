from flask import Flask, request, render_template, send_file
from azure.storage.blob import BlobServiceClient
import os
import json

app = Flask(__name__)

# Azure Storage Connection
with open(".\azure_keys.json") as keys_file:
    data = keys_file.read()
STORAGE_ACCOUNT_CONNECTION_STRING = json.load(data).AZURE_STORAGE_CONNECTION_STRING
blob_service_client = BlobServiceClient.from_connection_string(STORAGE_ACCOUNT_CONNECTION_STRING)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    """ Ladda upp en MP3-fil till Azure Blob Storage """
    file = request.files["file"]
    if file:
        blob_client = blob_service_client.get_blob_client(container="uploaded-audio", blob=file.filename)
        blob_client.upload_blob(file, overwrite=True)
        return "Fil uppladdad och transkriberas!", 200
    return "Ingen fil vald", 400

@app.route("/transcriptions", methods=["GET"])
def list_transcriptions():
    """ Lista alla färdiga transkriberingsfiler """
    container_client = blob_service_client.get_container_client("transcriptions")
    files = [blob.name for blob in container_client.list_blobs()]
    return {"transcriptions": files}

@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    """ Ladda ner en transkriberad fil """
    blob_client = blob_service_client.get_blob_client(container="transcriptions", blob=filename)
    file_path = f"/tmp/{filename}"
    
    with open(file_path, "wb")
