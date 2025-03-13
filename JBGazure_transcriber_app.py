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

@app.route("/", methods=["GET"])
def index():
    """ Render the upload/download page """
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    """ Ladda upp en MP3-fil till Azure Blob Storage """
    file = request.files["file"]
    if file:
        blob_client = blob_service_client.get_blob_client(container="uploaded-audio", blob=file.filename)
        blob_client.upload_blob(file, overwrite=True)
        return "File uploaded successfully and will be transcribed!", 200
    return "No file selected", 400

@app.route("/transcriptions", methods=["GET"])
def list_transcriptions():
    """ Lista alla färdiga transkriberingsfiler """
    container_client = blob_service_client.get_container_client("transcriptions")
    files = [blob.name for blob in container_client.list_blobs()]
    return {"transcriptions": files}

@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    """ Ladda ner en transkriberad fil från Azure Blob Storage """
    blob_client = blob_service_client.get_blob_client(container="transcriptions", blob=filename)
    
    try:
        # Ladda ner filen till en temporär plats
        file_path = f"/tmp/{filename}"  # Temporary location
        with open(file_path, "wb") as file:
            file.write(blob_client.download_blob().readall())

        # Skicka filen till användaren
        return send_file(file_path, as_attachment=True)

    except Exception as e:
        return f"Error downloading file: {str(e)}", 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
