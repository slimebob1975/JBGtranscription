# 📄 JBG Transkribering

**JBG Transkribering** is a browser-based audio transcription service built with **FastAPI**. It transcribes `.mp3` files locally with KB-Whisper and can optionally use OpenAI models for summaries and other text analysis.

The completed transcription and selected analyses are assembled as a **Microsoft Word (`.docx`) document**. The result is no longer displayed in the GUI. When processing is complete, the browser automatically downloads the DOCX. With encryption enabled, the DOCX package is created in memory and only AES-GCM-encrypted bytes are written to the server-side `results/` folder.

---

## 🚀 Features

- 🎙️ Upload `.mp3` audio files for transcription
- 🧠 Optional OpenAI-powered analysis:
  - Generate a short or extensive **summary**
  - Flag **suspected transcription errors**
  - Suggest **follow-up questions**
  - Attempt **speaker identification** (beta)
- 📝 Generate the complete result as a structured `.docx` document
- 🧩 Separate Word headings for each available output section
- ⬇️ Automatically download the DOCX when transcription and selected analyses are complete
- 🔑 Enter an OpenAI API key in the browser; it is not persisted server-side
- 🔒 Optional client-side encryption of the uploaded audio file
- 🔐 AES-256-GCM encryption of the server-side DOCX result when encryption is enabled
- 🌐 FastAPI-based frontend/backend suitable for local or Azure deployment

---

## 📝 DOCX Output

After transcription and all selected analysis steps have completed, the service creates a structured Word document containing:

1. **Rå transkribering**
2. **Transkribering med tidsstämplar**
3. **Sammanfattning** (optional)
4. **Transkription med markerade misstänkta fraser** (optional)
5. **Uppföljningsfrågor** (optional)
6. **Försök till identifiering av olika talare** (optional)

Each top-level output section uses a real Word heading style.

When encryption is enabled, the complete DOCX package is first generated in a `BytesIO` memory stream and then encrypted with AES-256-GCM before it is written to disk. The server-side file therefore has a name such as:

```plaintext
results/intervju_a1b2c3d4-e5f6-47a8-9012-3456789abcde.docx.encrypted
```

That file is **not** a readable Word document while it is on disk. When the job is complete, the browser automatically requests the result. The server decrypts the encrypted bytes only in memory and streams them back with the normal client filename:

```plaintext
intervju_a1b2c3d4-e5f6-47a8-9012-3456789abcde.docx
```

After the HTTP response has finished streaming, the server-side result file is deleted. If encryption is disabled, the temporary server-side result is a normal `.docx` file and is likewise deleted after it has been streamed.

> **Strict encrypted-at-rest operation:** set `ENCRYPTION_IS_OPTIONAL=0`. The GUI then forces encryption and the backend rejects upload requests that omit the encryption key. With this configuration, neither the uploaded audio nor the generated DOCX is stored as plaintext on server disk. Plaintext necessarily exists transiently in process memory while transcription, analysis, DOCX generation and download are performed.

---

## 🗂️ Project Structure

```plaintext
.
├── main.py                         # FastAPI backend entry point
├── requirements.txt                # Python dependencies
├── requirements_legacy.txt         # Unpinned dependency list
├── startup.sh                      # Startup script for Azure App Service
├── src/
│   ├── JBGtranscriber.py           # Transcription, analysis and DOCX generation
│   ├── JBGSecureFileHandler.py     # AES-GCM handling for audio and DOCX results
│   └── JBGLogger.py                # Logging
├── policy/
│   └── prompt_policy.json          # OpenAI prompt configuration
├── static/
│   ├── index.html                  # Web frontend
│   ├── styles/styles.css           # Frontend styling
│   └── javascript/script.js        # Upload and status polling
├── uploads/                        # Temporary uploaded audio files
└── results/                        # Temporary encrypted/plain DOCX result files awaiting download
```

---

## 🛠️ Prerequisites

- Python 3.9+
- An OpenAI API key if any OpenAI-based analysis is selected
- Git and pip

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/jbg-transkribering.git
   cd jbg-transkribering
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   ```

   Linux/macOS:

   ```bash
   source venv/bin/activate
   ```

   Windows PowerShell:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

`python-docx` is used to generate the Word result documents.

---

## ▶️ Running Locally

```bash
uvicorn main:app --reload
```

Then open:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🌐 Web Interface

### How to use

1. Select an `.mp3` file.
2. Enter an OpenAI API key if you want to use the optional analysis functions.
3. Select an OpenAI model.
4. Select the desired analysis options.
5. Click **Ladda upp**.
6. Follow the processing status in the GUI.
7. When transcription and all selected analyses are complete, the `.docx` file is downloaded automatically.

The transcription text itself is never inserted into the GUI.

---

## 🔐 Encryption

The web interface can encrypt the uploaded audio and the generated DOCX result with the same browser-generated AES-256-GCM key:

1. An **AES-256-GCM** key is generated locally in the browser.
2. The `.mp3` content is encrypted in the browser before upload.
3. The encrypted audio and base64-encoded key are sent to the server over the application connection.
4. The server decrypts the audio **in memory** for transcription.
5. After transcription and all requested analyses, `python-docx` builds the Word package in an in-memory `BytesIO` stream.
6. The in-memory DOCX bytes are encrypted with AES-256-GCM before being written to the `results/` directory. No plaintext DOCX temporary file is created in encrypted mode.
7. When processing is complete, the frontend automatically calls the download endpoint. The encryption key is sent in POST form data rather than in the URL.
8. The server decrypts the result **in memory** and streams a normal `.docx` response to the browser.
9. After the response has finished streaming, the server deletes the stored result file.
10. The temporary uploaded audio file is also deleted after processing.

### Enforcing encryption

`ENCRYPTION_IS_OPTIONAL` controls whether the user may disable encryption:

```env
ENCRYPTION_IS_OPTIONAL=0
```

With the value `0`, encryption is mandatory in the GUI and the backend rejects unencrypted upload requests. Use this setting when encrypted-at-rest handling is a hard requirement.

With the value `1`, the user may turn encryption off. In that mode the result is temporarily stored as a normal DOCX until it has been streamed to the browser and deleted.

### What "encrypted on the server" means

With encryption enabled, audio and DOCX result files are encrypted **at rest on server disk**. The application must still hold plaintext transiently in RAM while it transcribes the audio, performs text analysis, constructs the DOCX and streams the downloaded document. This is the same basic security model as the previous encrypted TXT workflow.

The OpenAI API key is kept in the browser's `localStorage` for convenience by the current frontend. It is sent with the transcription request but is not intentionally persisted by the server application.

---

## 🧠 Optional AI Analysis

Each checkbox triggers a separate analysis step:

- **Summary**: produces a short or extensive summary.
- **Suspected transcription errors**: identifies phrases that may have been transcribed incorrectly.
- **Follow-up questions**: generates questions based on the transcription.
- **Speaker identification (beta)**: attempts to separate or identify speakers based on the transcribed content.

Only selected/available analyses are included in the DOCX output.

---

## 🧠 Prompt Customization via `prompt_policy.json`

OpenAI-related prompts are stored in:

```plaintext
policy/prompt_policy.json
```

Supported keys include:

| Key | Used for |
|---|---|
| `short_summary` | Short summary instructions |
| `extensive_summary` | Detailed summary instructions |
| `suspicious_phrases` | Suspected transcription-error detection |
| `follow_up_questions` | Follow-up question generation |
| `speaker_diarization` | Speaker identification/segmentation |

The prompt policy is loaded dynamically for each transcriber instance.

> This implementation is currently optimized for Swedish transcriptions.

---

## 🧪 Command-Line Transcription Tool

`src/JBGtranscriber.py` can also be run independently.

### Usage

```bash
python src/JBGtranscriber.py [input_path] [output_folder] [cpu|gpu] [openai_api_key] [model] [summary_style]
```

### Example

```bash
python src/JBGtranscriber.py ./audio ./results cpu sk-xxxxxx gpt-5.2 extensive
```

The command accepts either a single `.mp3` file or a directory containing `.mp3` files. Each input file produces a corresponding `.docx` file in the selected output directory.

---

## ☁️ Deployment to Azure App Service

This project can be deployed to Azure App Service.

```bash
az login
az group create --name jbg-rg --location westeurope
az webapp up --name jbg-transkribering --resource-group jbg-rg --sku B1 --runtime "PYTHON:3.9"
az webapp config set --name jbg-transkribering --resource-group jbg-rg --startup-file startup.sh
```

The application uses its local `results/` directory as temporary result storage while a completed DOCX is waiting to be downloaded. In encrypted mode those files contain AES-GCM ciphertext rather than a readable Word package. Files are deleted after their download response has finished streaming, although stale files can remain after interrupted jobs or failed downloads and should be covered by an operational cleanup policy.

---

## 🔐 Security and Retention Notes

- Uploaded audio is temporary and is deleted after processing.
- With encryption enabled, uploaded audio is encrypted at rest.
- With encryption enabled, the DOCX is generated in memory and only AES-GCM ciphertext is stored in `results/`.
- The result is decrypted only to an in-memory stream for download.
- The encryption key is not included in status URLs; the download request sends it as POST form data.
- The AES key exists only in the current browser page session. Reloading or closing the page before the automatic download loses that key; an encrypted result left on the server would then require cleanup rather than recovery through the GUI.
- The stored result file is deleted after the download response has finished streaming.
- A server cannot reliably know that the browser ultimately saved a file to the user's filesystem; it can only know that the full HTTP response was delivered.
- Interrupted processing or failed downloads can leave stale files in `uploads/` or `results/`; retain a separate cleanup policy for those cases.
- Set `ENCRYPTION_IS_OPTIONAL=0` when plaintext server-side result files must never be permitted by application configuration.
- The OpenAI API key is not intentionally stored server-side.

---

## 🧼 Cleanup

Normal successful web runs delete the temporary uploaded audio and remove the result file after it has been streamed to the browser. Operational cleanup is still recommended for interrupted jobs and failed downloads.

Example manual cleanup in PowerShell:

```powershell
Remove-Item .\uploads\* -Force
Remove-Item .\results\*.docx.encrypted -Force
Remove-Item .\results\*.docx -Force
Remove-Item .\results\*.tmp -Force
```

For production use, prefer a retention rule that only removes stale files older than an appropriate threshold rather than deleting active jobs.

---

## 📜 License

MIT License — see `LICENSE` if included in the repository.

---

## 🤝 Contributions

Pull requests are welcome. Please open an issue first to discuss substantial changes.

---

## 📧 Contact

For questions, contact:  
📨 robert.granat@iaf.se  
🌍 [www.iaf.se](https://www.iaf.se)
