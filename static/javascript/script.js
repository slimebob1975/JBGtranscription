// ✅ On page load: enable/disable fields based on API key presence
document.addEventListener("DOMContentLoaded", () => {
    const savedKey = localStorage.getItem("openai_api_key");
    if (savedKey) {
        document.getElementById("apiKey").value = savedKey;
    }

    const apiKeyInput = document.getElementById("apiKey");
    const checkboxes = [
        document.getElementById("optSummary"),
        document.getElementById("optSuspicious"),
        document.getElementById("optQuestions"),
        document.getElementById("optSpeakers"),
    ];

    apiKeyInput.addEventListener("input", () => {
        const hasKey = apiKeyInput.value.trim().length > 0;
        checkboxes.forEach(cb => cb.disabled = !hasKey);
    });

    // Summary checkbox: toggle radio buttons for style
    const summaryCheckbox = document.getElementById("optSummary");
    const summaryOptions = document.getElementById("summaryOptions");

    summaryCheckbox.addEventListener("change", () => {
        const show = summaryCheckbox.checked;
        summaryOptions.style.display = show ? "block" : "none";
        document.querySelectorAll('input[name="summaryStyle"]').forEach(rb => rb.disabled = !show);
    });
});

// ✅ Upload handler
async function uploadFile() {
    const fileInput = document.getElementById("audioFile");
    if (fileInput.files.length === 0) {
        alert("Please select a file.");
        return;
    }

    const file = fileInput.files[0];
    if (file.type !== "audio/mpeg") {
        alert("Only MP3 files are allowed.");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);
    formData.append("api_key", document.getElementById("apiKey").value.trim());
    formData.append("model", document.getElementById("modelSelect").value);
    formData.append("summarize", document.getElementById("optSummary").checked);
    formData.append("summary_style", document.querySelector('input[name="summaryStyle"]:checked')?.value || "short");
    formData.append("suspicious", document.getElementById("optSuspicious").checked);
    formData.append("questions", document.getElementById("optQuestions").checked);
    formData.append("speakers", document.getElementById("optSpeakers").checked);

    // Disable inputs
    fileInput.disabled = true;
    document.getElementById("button").disabled = true;
    document.getElementById("optSummary").disabled = true;
    document.getElementById("optSuspicious").disabled = true;
    document.getElementById("optQuestions").disabled = true;
    document.getElementById("optSpeakers").disabled = true;

    // Store API key locally
    localStorage.setItem("openai_api_key", document.getElementById("apiKey").value.trim());

    let response = await fetch("/upload/", {
        method: "POST",
        body: formData
    });

    let result = await response.json();
    console.log(result);

    if (result.file_id) {
        document.getElementById("status").innerText = "Fil uppladdad. Processar...";
        checkStatus(result.file_id);
    }
}

// ✅ Polling for transcription status
async function checkStatus(file_id) {
    let spinner = document.getElementById("spinner-container");
    spinner.style.display = "block";

    setTimeout(async () => {
        let response = await fetch("/transcription/" + file_id);
        if (response.ok) {
            spinner.style.display = "none";

            let result = await response.json();
            document.getElementById("status").innerText = "Transkribering avslutad. Tryck Ctrl-Shift-R för att köra igen!";
            document.getElementById("result").value = result.transcription;
            document.getElementById("result").style.display = "inline";

            document.getElementById("downloadLink").href = "/download/" + file_id;
            document.getElementById("downloadLink").style.display = "inline";
        } else {
            checkStatus(file_id);
        }
    }, 5000);
}
