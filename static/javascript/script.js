async function uploadFile() {
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
    formData.append("file", fileInput.files[0]);

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
            document.getElementById("status").innerText = "Transkribering avslutad!";
            
            
            document.getElementById("result").value = result.transcription;  
            document.getElementById("result").style.display = "inline"; 

            document.getElementById("downloadLink").href = "/download/" + file_id;
            document.getElementById("downloadLink").style.display = "inline";

        } else {
            checkStatus(file_id);
        }
    }, 5000);
}