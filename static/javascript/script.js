async function uploadFile() {
    
    document.addEventListener("DOMContentLoaded", () => {
        const apiKeyInput = document.getElementById("apiKey");
        const checkboxes = [
            document.getElementById("optSummary"),
            document.getElementById("optSuspicious"),
            document.getElementById("optQuestions")
        ];
    
        apiKeyInput.addEventListener("input", () => {
            const hasKey = apiKeyInput.value.trim().length > 0;
            checkboxes.forEach(cb => cb.disabled = !hasKey);
        });
    });
    
    let fileInput = document.getElementById("audioFile");
    if (fileInput.files.length === 0) {
        alert("Please select a file.");
        return;
    }

    // Restrict to MP3 files only
    let file = fileInput.files[0];
    if (file.type !== "audio/mpeg") {  // audio/mpeg = MP3 file type
        alert("Only MP3 files are allowed.");
        return;
    }
    
    let formData = new FormData();
    // Add file
    formData.append("file", fileInput.files[0]);

    // Add API key
    formData.append("api_key", document.getElementById("apiKey").value.trim());

    // Add model selection
    formData.append("model", document.getElementById("modelSelect").value);

    // Add checkbox values
    formData.append("summarize", document.getElementById("optSummary").checked);
    formData.append("suspicious", document.getElementById("optSuspicious").checked);
    formData.append("questions", document.getElementById("optQuestions").checked);
    formData.append("speakers", document.getElementById("optSpeakers").checked);


    // Disable forms for this time
    document.getElementById("audioFile").disabled = true;
    document.getElementById("button").disabled = true;
    document.getElementById("optSummary").disabled = true;
    document.getElementById("optSuspicious").disabled = true;
    document.getElementById("optQuestions").disabled = true;
    document.getElementById("optSpeakers").disabled = true;

    // Do upload and wait for response
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