// ✅ On page load: enable/disable fields based on API key presence
let globalEncryptionKeyBase64 = null;
let encryptionEnabled = true;

fetch("/config")
  .then(res => res.json())
  .then(data => {
    document.title = data.title;
    document.getElementById("app-title").innerText = data.title;
    const checkbox = document.getElementById('enableEncryption');
    if (data.encryption_is_optional === "1") {
      checkbox.disabled = false;
    } else {
      checkbox.checked = true;
      checkbox.disabled = true;
    }
  });

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

    // Show currently logged-in Azure AD user (if available)
    fetch("/me")
    .then(res => res.json())
    .then(data => {
        const userLabel = document.getElementById("userDisplay");
        if (data.user && userLabel) {
            userLabel.textContent = `Inloggad som: ${data.user}`;
        }
    })
    .catch(err => {
        console.warn("Kunde inte hämta användarinformation:", err);
    });
});

// ✅ Upload handler
async function uploadFile() {

    // Take care of data
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

    // Update status
    document.getElementById("status").innerText = "Filen laddas upp...-v.g. vänta!";

     // Disable inputs
    document.getElementById("enableEncryption").disabled = true;
    document.getElementById("apiKey").disabled = true;
    document.getElementById("modelSelect").disabled = true;
    document.getElementById("optSummary").disabled = true;
    document.getElementById("optSuspicious").disabled = true;
    document.getElementById("optQuestions").disabled = true;
    document.getElementById("optSpeakers").disabled = true;
    document.getElementById("button").disabled = true;

    // Prepare data to send for backend
    let formData = new FormData();

    // Should we use encryption or not?
    const encryptEnabled = document.getElementById("enableEncryption").checked;
    if (encryptEnabled) {
        // 🔑 Generera AES-nyckel
        const key = await window.crypto.subtle.generateKey(
            { name: "AES-GCM", length: 256 },
            true,
            ["encrypt", "decrypt"]
        );

        // 🔄 Konvertera nyckel till base64 för att skicka till backend
        const rawKey = await crypto.subtle.exportKey("raw", key);
        const keyBase64 = btoa(String.fromCharCode(...new Uint8Array(rawKey)));

        // 🔐 Kryptera filinnehåll
        const iv = window.crypto.getRandomValues(new Uint8Array(12));
        const arrayBuffer = await file.arrayBuffer();
        const encrypted = await crypto.subtle.encrypt(
            { name: "AES-GCM", iv: iv },
            key,
            arrayBuffer
        );

        // Skapa ny Blob för krypterad fil
        const encryptedBlob = new Blob([iv, encrypted], { type: "application/octet-stream" });

        formData.append("file", encryptedBlob, file.name + ".enc");
        formData.append("encryption_key", keyBase64);

        globalEncryptionKeyBase64 = keyBase64;  // 👈 Gör nyckeln tillgänglig för checkStatus
        encryptionEnabled = true;
    } else {
        formData.append("file", file);
        formData.append("encryption_key", ""); // eller null eller "none"
        globalEncryptionKeyBase64 = null;
        encryptionEnabled = false;
    }

    formData.append("api_key", document.getElementById("apiKey").value.trim());
    formData.append("model", document.getElementById("modelSelect").value);
    formData.append("summarize", document.getElementById("optSummary").checked);
    formData.append("summary_style", document.querySelector('input[name="summaryStyle"]:checked')?.value || "short");
    formData.append("suspicious", document.getElementById("optSuspicious").checked);
    formData.append("questions", document.getElementById("optQuestions").checked);
    formData.append("speakers", document.getElementById("optSpeakers").checked);

    fileInput.disabled = true;

    // Check Formdata for errors.
    for (const [key, value] of formData.entries()) {
        // Om värdet är en Blob (fil), visa namn och typ
        if (value instanceof Blob) {
            console.log(`${key}: [Blob] filename=${value.name}, type=${value.type}, size=${value.size}`);
        } else {
            console.log(`${key}: ${value}`);
        }
    }

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

// ✅ Polling for transcription status – with live step updates
async function checkStatus(file_id) {
    const spinner = document.getElementById("spinner-container");
    spinner.style.display = "block";

    setTimeout(async () => {
        try {
            const url = `/transcription/${file_id}?encryption_key=${encodeURIComponent(globalEncryptionKeyBase64 || "")}`;
            const response = await fetch(url);
            console.info("HTTP status:", response.status);

            // 1) Jobbet pågår → 202 Accepted
            if (response.status === 202) {
                let result;
                try {
                    result = await response.json();
                } catch (e) {
                    console.error("Kunde inte tolka 202-svar från servern:", e);
                    document.getElementById("status").innerText =
                        "Tekniskt fel vid statuskontroll. Försöker igen...";
                    checkStatus(file_id);
                    return;
                }

                document.getElementById("status").innerText =
                    result.status || "Processar...";
                // Fortsätt polla
                checkStatus(file_id);
                return;
            }

            // 2) Andra felaktiga HTTP-statusar (4xx, 5xx)
            if (!response.ok) {
                const text = await response.text().catch(() => "");
                console.error(
                    "Serverfel vid /transcription:",
                    response.status,
                    text
                );
                spinner.style.display = "none";
                document.getElementById("status").innerText =
                    `Serverfel (${response.status}). Ett fel uppstod vid transkriberingen.`;
                return;
            }

            // 3) 200 OK → nu ska vi ha ett färdigt resultat
            let result;
            try {
                result = await response.json();
            } catch (e) {
                console.error("Kunde inte tolka 200-svar från servern:", e);
                document.getElementById("status").innerText =
                    "Tekniskt fel vid statuskontroll. Försöker igen...";
                checkStatus(file_id);
                return;
            }

            // Extra säkerhet: om backend i framtiden skickar error/done här
            if (result.error) {
                spinner.style.display = "none";
                document.getElementById("status").innerText =
                    result.status || "Ett fel uppstod vid transkriberingen.";
                console.error("Transcription error from server:", result.error);
                return;
            }

            if (result.done === false) {
                document.getElementById("status").innerText =
                    result.status || "Processar...";
                checkStatus(file_id);
                return;
            }

            // Klar och lyckad
            if (result.done === true) {
                spinner.style.display = "none";

                const statusText = result.status || "Transkribering avslutad.";
                const decodedText = result.transcription || "";

                document.getElementById("status").innerText = statusText;
                document.getElementById("result").value = decodedText;
                document.getElementById("result").style.display = "inline";

                const textBlob = new Blob([decodedText], { type: "text/plain" });
                const urlBlob = URL.createObjectURL(textBlob);
                document.getElementById("downloadLink").href = urlBlob;
                document.getElementById("downloadLink").style.display = "inline";

                return;
            }

            // Om vi hamnar här är något oväntat
            console.warn("Oväntat svar från /transcription:", result);
            document.getElementById("status").innerText =
                "Oväntat svar från servern. Försöker igen...";
            checkStatus(file_id);
        } catch (err) {
            console.error("Fel vid statuskontroll:", err);
            document.getElementById("status").innerText =
                "Tekniskt fel vid statuskontroll. Försöker igen...";
            checkStatus(file_id);
        }
    }, 3000);
}
