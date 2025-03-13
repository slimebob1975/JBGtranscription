import os
import azure.functions as func
from azure.storage.blob import BlobServiceClient
import subprocess
import json

# Azure Blob Storage inställningar
with open(".\azure_keys.json") as keys_file:
    data = keys_file.read()
STORAGE_ACCOUNT_CONNECTION_STRING = json.load(data).AZURE_STORAGE_CONNECTION_STRING
BLOB_SERVICE_CLIENT = BlobServiceClient.from_connection_string(STORAGE_ACCOUNT_CONNECTION_STRING)

def transcribe_audio(file_path, output_path):
    """ Kör transkriberingsscriptet som ett subprocess-anrop """
    command = ["python.exe", "JBGtranscriber.py", file_path, output_path, "gpu"]
    subprocess.run(command, check=True)

def main(blob: func.InputStream):
    """ Funktion som triggas när en MP3 laddas upp """
    file_name = blob.name.split("/")[-1]
    local_file_path = f"/tmp/{file_name}"
    
    # Ladda ner filen från blob storage
    with open(local_file_path, "wb") as f:
        f.write(blob.read())

    # Kör transkriberingsscriptet
    output_file = f"/tmp/{file_name}.txt"
    transcribe_audio(local_file_path, output_file)

    # Ladda upp transkriberingen till transcriptions-container
    transcriptions_container = BLOB_SERVICE_CLIENT.get_container_client("transcriptions")
    with open(output_file, "rb") as f:
        transcriptions_container.upload_blob(name=f"{file_name}.txt", data=f, overwrite=True)

    print(f"Transkribering klar för: {file_name}")
