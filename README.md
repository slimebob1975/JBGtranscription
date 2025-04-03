# 📄 JBG Transkribering

**JBG Transkribering** is a secure, browser-based audio transcription tool built with **FastAPI** and enhanced with **optional OpenAI-powered analysis**. It allows users to upload `.mp3` files and receive clean text transcriptions, with the option to generate summaries, flag suspicious phrases, and suggest follow-up questions.

> ✨ The frontend mimics the visual style of [iaf.se](https://www.iaf.se) and requires **no installation or login** to use.

---

## 🚀 Features

- 🎙️ Upload `.mp3` audio files for transcription
- 🧠 Optional AI enhancements via OpenAI:
  - Generate **summaries**
  - Flag **suspicious phrases**
  - Suggest **follow-up questions**
- 🔑 Enter your own **OpenAI API key** in the browser — no key is stored or logged on the server
- 🧩 Choose from multiple OpenAI models (e.g., `gpt-4o`, `gpt-4`, `gpt-3.5-turbo`)
- 📄 Transcription output shown in a scrollable field and available for download
- 🔒 Local file processing; secure and self-contained
- 🌐 Azure-ready and easy to deploy

---

## 🗂️ Project Structure

```plaintext
.
├── main.py                    # FastAPI backend entry point
├── requirements.txt           # Python dependencies
├── startup.sh                 # Startup script for Azure App Service
├── JBGtranscriber.py          # Core logic: transcription + OpenAI-based analysis
├── static/
│   ├── index.html             # Web frontend
│   ├── styles.css             # IAF-inspired styling
│   └── script.js              # JS logic for file upload and form controls
├── uploads/                   # Uploaded audio files (local use)
└── results/                   # Transcribed text results
```

---

## 🛠️ Prerequisites

- Python 3.9+
- An OpenAI API key
- Git & pip

---

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-org/jbg-transkribering.git
   cd jbg-transkribering
   ```

2. **Create virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running Locally

```bash
uvicorn main:app --reload
```

Then open your browser at:  
➡️ [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🌐 Web Interface (Frontend)

### How to Use:
1. **Select an `.mp3` file**
2. **Enter your OpenAI API key**
3. **Choose model** (e.g. `gpt-4o`)
4. **Select optional AI features:**
   - ✔️ Summary
   - ✔️ Suspicious phrase detection
   - ✔️ Follow-up questions
5. **Click "Ladda upp"**

📄 The transcription will appear when ready, with a download link to the full `.txt` file.

---

Absolutely — here's the updated section of the `README.md` with a clear note on language support:

---

## 🧠 What the AI Does (Optional Analysis)

Each optional AI checkbox triggers a different OpenAI-powered feature:
- **Summary**: Condenses the transcription into key points.
- **Suspicious Phrases**: Flags content that could be controversial or sensitive.
- **Follow-up Questions**: Suggests possible questions to dig deeper into the discussion.
- **Try to identify different speakers**: Suggests different speakers in the transcribed material.

All results are included in the output `.txt`.

> 💬 **Note:** This implementation is currently optimized for **Swedish transcriptions**.  
> To support other languages, you can modify the `transcribe()` method in the `JBGtranscriber` class to set the desired transcription language and prompt instructions.

---

## 🧪 Command-Line Transcription Tool

The `JBGtranscriber.py` script can be used independently of the web interface.

### Usage:

```bash
python JBGtranscriber.py [input_path] [output_folder] [cpu|gpu] [openai_api_key] (optional: model)
```

### Example:

```bash
python JBGtranscriber.py ./audio ./results cpu sk-xxxxxx gpt-4o
```

### Features:
- Accepts **single files or folders** with `.mp3` files
- Processes with or without OpenAI enhancements
- Saves clean `.txt` outputs to the specified directory

---

## ☁️ Deployment to Azure App Service

This project is Azure-ready.

### 1. Login and Create Resource Group
```bash
az login
az group create --name jbg-rg --location westeurope
```

### 2. Deploy the App
```bash
az webapp up --name jbg-transkribering --resource-group jbg-rg --sku B1 --runtime "PYTHON:3.9"
```

### 3. Set Startup Script
```bash
az webapp config set --name jbg-transkribering --resource-group jbg-rg --startup-file startup.sh
```

📦 No need to configure the OpenAI key on the server — the user provides it securely from the frontend.

---

## 🔐 Security

- API key is entered in the browser only
- API key is never stored or logged
- Transcriptions are not shared or retained outside the session
- Suitable for private or internal use

---

## 🧼 Cleanup

Temporary files (`uploads/`, `results/`) can be cleared automatically or periodically.  
To retain only the latest transcription:

```python
# Inside main.py
old_transcripts = sorted(
    results_path.glob("*.mp3.txt"),
    key=lambda f: f.stat().st_ctime
)[:-1]
for file in old_transcripts:
    file.unlink()
```

---

## 📜 License

MIT License — see [LICENSE](./LICENSE)

---

## 🤝 Contributions

Pull requests are welcome! Please open an issue first to discuss what you’d like to change.

---

## 📧 Contact

For questions, contact:  
📨 robert.granat@iaf.se  
🌍 [www.iaf.se](https://www.iaf.se)

---

Would you like a **Swedish version** of the README or parts of it as well?  
Let me know if you also want to include **badges**, **Docker instructions**, or a GitHub Actions CI setup.