// ✅ On page load: enable/disable fields based on API key presence
let globalEncryptionKeyBase64 = "";

// --- Editable summary instruction -------------------------------------------
// The two default texts ("Enkel" and "Utförlig") are fetched from the backend so
// that policy/prompt_policy.json stays the single source of truth. The user may
// edit the text freely; an edited text is remembered between visits.

const SUMMARY_PROMPT_STORAGE_KEY = "jbg_summary_prompt";
const SUMMARY_STYLE_STORAGE_KEY = "jbg_summary_style";

// Option ids retired in an earlier version. Browsers still hold these, so a
// stored value is mapped rather than silently falling back to the default.
const LEGACY_SUMMARY_STYLE_IDS = { short: "enkel", extensive: "utforlig" };

let summaryOptions = [];              // [{id, label, description, default, prompt}]
let summaryPromptMaxLength = 20000;
let summaryOptionsLoaded = false;

function currentSummaryStyle() {
    return document.getElementById("summaryStyle").value || "";
}

function summaryOptionById(id) {
    return summaryOptions.find(o => o.id === id) || null;
}

function defaultSummaryPromptFor(id) {
    const option = summaryOptionById(id);
    return option ? option.prompt : "";
}

function summaryStyleLabel(id) {
    const option = summaryOptionById(id);
    return option ? option.label : id;
}

function isSummaryPromptEdited() {
    const textarea = document.getElementById("summaryPrompt");
    return textarea.value.trim() !== defaultSummaryPromptFor(currentSummaryStyle()).trim();
}

function matchesAnySummaryDefault(text) {
    const trimmed = (text || "").trim();
    return summaryOptions.some(o => o.prompt.trim() === trimmed);
}

function refreshSummaryStyleDescription() {
    // The description is carried by the tooltip on the select and on each
    // option; there is no separate visible line.
    const option = summaryOptionById(currentSummaryStyle());
    const select = document.getElementById("summaryStyle");
    const description = option ? option.description : "";
    if (description) {
        select.title = description;
    } else {
        select.removeAttribute("title");
    }
}

function refreshSummaryPromptState() {
    const state = document.getElementById("summaryPromptState");
    const resetButton = document.getElementById("resetSummaryPrompt");

    if (!summaryOptionsLoaded) {
        state.textContent = "Standardtexterna kunde inte hämtas. Skriv en egen instruktion, annars används serverns standardtext.";
        state.hidden = false;
        resetButton.hidden = true;
        return;
    }

    // Nothing is shown while a default text is in place - that is the normal
    // case and needs no label. An edited instruction is still called out,
    // since it is the state the user could otherwise forget they are in.
    if (isSummaryPromptEdited()) {
        state.textContent = "Egen text används.";
        state.hidden = false;
        resetButton.hidden = false;
    } else {
        state.textContent = "";
        state.hidden = true;
        resetButton.hidden = true;
    }
    const subtitle = document.getElementById("summaryEditorSubtitle");
    if (subtitle) {
        subtitle.textContent = `Utgår från: ${summaryStyleLabel(currentSummaryStyle())}`;
    }
}

function applySummaryDefault(id) {
    document.getElementById("summaryPrompt").value = defaultSummaryPromptFor(id);
    storeSummaryPrompt();
    refreshSummaryStyleDescription();
    refreshSummaryPromptState();
}

function storeSummaryPrompt() {
    try {
        localStorage.setItem(SUMMARY_PROMPT_STORAGE_KEY, document.getElementById("summaryPrompt").value);
        localStorage.setItem(SUMMARY_STYLE_STORAGE_KEY, currentSummaryStyle());
    } catch (err) {
        console.warn("Kunde inte spara sammanfattningsinstruktionen lokalt:", err);
    }
}

function populateSummaryStyleSelect() {
    const select = document.getElementById("summaryStyle");
    select.innerHTML = "";
    summaryOptions.forEach(option => {
        const el = document.createElement("option");
        el.value = option.id;
        el.textContent = option.label;
        if (option.description) el.title = option.description;
        if (option.default) el.selected = true;
        select.appendChild(el);
    });
}

// --- Floating editor ---------------------------------------------------------
// The instruction is edited in a <dialog> so that showing it never changes the
// height of the page. Native showModal() is used for the focus trap, the
// backdrop and Esc; only its default centring is overridden, because that
// centres on the iframe's own viewport rather than what the user can see.

let summaryPromptSnapshot = null;   // text as it was when the editor opened
let latestParentPageInfo = null;    // from iframe-resizer, when embedded

function trackParentPageInfo() {
    // getPageInfo reports the parent's scroll position and viewport, so the
    // editor can open where the user is actually looking. Absent or silent
    // when not embedded, in which case the trigger anchor is used alone.
    try {
        if (window.parentIFrame && typeof window.parentIFrame.getPageInfo === "function") {
            window.parentIFrame.getPageInfo(info => { latestParentPageInfo = info; });
        }
    } catch (err) {
        console.warn("Kunde inte läsa förälderns sidinformation:", err);
    }
}

function positionSummaryEditor(trigger) {
    // Called after the dialog is shown, so its real height can be measured.
    const dialog = document.getElementById("summaryEditor");
    const dialogHeight = dialog.offsetHeight || 460;
    const info = latestParentPageInfo;

    // Fallback: anchor above the link the user just clicked. It is on screen by
    // definition, so this is safe when nothing else is known.
    let top = trigger.getBoundingClientRect().top + window.scrollY - 120;

    if (info && typeof info.scrollTop === "number" && typeof info.offsetTop === "number") {
        // Embedded, and the parent has reported its scroll position: centre on
        // the part of the iframe the user can actually see.
        const visibleTop = info.scrollTop - info.offsetTop;
        const visibleHeight = info.clientHeight || info.windowHeight || 0;
        if (visibleHeight > 0) {
            top = visibleTop + (visibleHeight - dialogHeight) / 2;
        }
    } else if (!window.parentIFrame) {
        // Standalone: innerHeight is the real window, so centre on it.
        top = window.scrollY + (window.innerHeight - dialogHeight) / 2;
    }

    dialog.style.top = `${Math.round(Math.max(8, top))}px`;
}

function openSummaryEditor() {
    const dialog = document.getElementById("summaryEditor");
    const textarea = document.getElementById("summaryPrompt");
    const trigger = document.getElementById("openSummaryEditor");

    summaryPromptSnapshot = textarea.value;

    if (typeof dialog.showModal === "function") {
        dialog.showModal();
    } else {
        dialog.setAttribute("open", "");   // very old browsers: no modality
    }
    positionSummaryEditor(trigger);
    textarea.focus();
    textarea.setSelectionRange(0, 0);
    textarea.scrollTop = 0;
}

function closeSummaryEditor(apply) {
    const dialog = document.getElementById("summaryEditor");
    const textarea = document.getElementById("summaryPrompt");

    if (!apply && summaryPromptSnapshot !== null) {
        textarea.value = summaryPromptSnapshot;   // Avbryt and Esc revert
    }
    summaryPromptSnapshot = null;

    storeSummaryPrompt();
    refreshSummaryPromptState();

    if (dialog.open && typeof dialog.close === "function") {
        dialog.close();
    } else {
        dialog.removeAttribute("open");
    }
}

function setUpSummaryEditorDialog() {
    const dialog = document.getElementById("summaryEditor");

    document.getElementById("openSummaryEditor")
        .addEventListener("click", openSummaryEditor);
    document.getElementById("applySummaryEditor")
        .addEventListener("click", () => closeSummaryEditor(true));
    document.getElementById("cancelSummaryEditor")
        .addEventListener("click", () => closeSummaryEditor(false));

    // Esc means the same here as everywhere else: discard and close.
    dialog.addEventListener("cancel", event => {
        event.preventDefault();
        closeSummaryEditor(false);
    });

    // Clicking the backdrop is treated as Avbryt.
    dialog.addEventListener("click", event => {
        if (event.target === dialog) closeSummaryEditor(false);
    });

    trackParentPageInfo();
}

function setUpSummaryPromptEditor() {
    const textarea = document.getElementById("summaryPrompt");
    const resetButton = document.getElementById("resetSummaryPrompt");
    const select = document.getElementById("summaryStyle");

    textarea.addEventListener("input", () => {
        if (textarea.value.length > summaryPromptMaxLength) {
            textarea.value = textarea.value.slice(0, summaryPromptMaxLength);
        }
        storeSummaryPrompt();
        refreshSummaryPromptState();
    });

    resetButton.addEventListener("click", () => {
        applySummaryDefault(currentSummaryStyle());
        textarea.focus();
    });

    setUpSummaryEditorDialog();

    // Remembered so the selection can be restored if the user cancels.
    let previousStyle = null;

    select.addEventListener("focus", () => { previousStyle = select.value; });

    select.addEventListener("change", () => {
        const newStyle = select.value;
        const edited = !matchesAnySummaryDefault(textarea.value) && textarea.value.trim() !== "";

        if (edited) {
            const proceed = confirm(
                `Din egen text ersätts av standardtexten för ${summaryStyleLabel(newStyle)}. Vill du fortsätta?`
            );
            if (!proceed) {
                if (previousStyle !== null) select.value = previousStyle;
                refreshSummaryStyleDescription();
                refreshSummaryPromptState();
                return;
            }
        }

        previousStyle = newStyle;
        applySummaryDefault(newStyle);
    });

    fetch("/summary_prompts")
        .then(res => res.json())
        .then(data => {
            summaryOptions = Array.isArray(data.options) ? data.options : [];
            summaryPromptMaxLength = data.max_length || summaryPromptMaxLength;
            summaryOptionsLoaded = summaryOptions.length > 0;

            if (!summaryOptionsLoaded) {
                refreshSummaryPromptState();
                return;
            }

            populateSummaryStyleSelect();

            let savedStyle = localStorage.getItem(SUMMARY_STYLE_STORAGE_KEY);
            if (savedStyle && LEGACY_SUMMARY_STYLE_IDS[savedStyle]) {
                savedStyle = LEGACY_SUMMARY_STYLE_IDS[savedStyle];
            }
            if (savedStyle && summaryOptionById(savedStyle)) {
                select.value = savedStyle;
            }
            previousStyle = select.value;

            // A stored text that matches one of the defaults is not a custom
            // instruction; it is just the default the user last looked at.
            const savedPrompt = localStorage.getItem(SUMMARY_PROMPT_STORAGE_KEY);
            const hasCustomText =
                savedPrompt !== null &&
                savedPrompt.trim() !== "" &&
                !matchesAnySummaryDefault(savedPrompt);

            textarea.value = hasCustomText ? savedPrompt : defaultSummaryPromptFor(select.value);

            refreshSummaryStyleDescription();
            refreshSummaryPromptState();
        })
        .catch(err => {
            console.warn("Kunde inte hämta standardtexter för sammanfattning:", err);
            summaryOptionsLoaded = false;
            refreshSummaryPromptState();
        });
}

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

    // Summary checkbox: reveal the style choice and the editable instruction
    const summaryCheckbox = document.getElementById("optSummary");
    const summaryOptions = document.getElementById("summaryOptions");

    summaryCheckbox.addEventListener("change", () => {
        const show = summaryCheckbox.checked;
        summaryOptions.hidden = !show;
        document.getElementById("summaryStyle").disabled = !show;
        document.getElementById("openSummaryEditor").disabled = !show;
    });

    setUpSummaryPromptEditor();

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
    document.getElementById("openSummaryEditor").disabled = true;
    document.getElementById("summaryStyle").disabled = true;
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
        globalEncryptionKeyBase64 = keyBase64;

    } else {
        formData.append("file", file);
        formData.append("encryption_key", "");
        globalEncryptionKeyBase64 = "";
    }

    formData.append("api_key", document.getElementById("apiKey").value.trim());
    formData.append("model", document.getElementById("modelSelect").value);
    const summarizeChecked = document.getElementById("optSummary").checked;
    formData.append("summarize", summarizeChecked);
    formData.append("summary_style", currentSummaryStyle());
    // Empty means "use the server-side default for the chosen style".
    formData.append(
        "summary_prompt",
        summarizeChecked ? document.getElementById("summaryPrompt").value.trim() : ""
    );
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

async function downloadResult(file_id, filename) {
    const formData = new FormData();
    formData.append("encryption_key", globalEncryptionKeyBase64 || "");

    const response = await fetch(`/download/${file_id}`, {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        const text = await response.text().catch(() => "");
        throw new Error(`Nedladdningen misslyckades (${response.status}): ${text}`);
    }

    // The server decrypts encrypted results only in memory and streams the DOCX.
    const blob = await response.blob();
    const blobUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = blobUrl;
    link.download = filename || "transkribering.docx";
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
}

// ✅ Polling for transcription status – with live step updates
async function checkStatus(file_id) {
    const spinner = document.getElementById("spinner-container");
    spinner.style.display = "block";

    setTimeout(async () => {
        try {
            const response = await fetch(`/transcription/${file_id}`);
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

            // 3) 200 OK → jobbet är färdigt och DOCX-filen är sparad på servern
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

            // Klar och lyckad: hämta DOCX automatiskt.
            if (result.done === true) {
                spinner.style.display = "none";
                document.getElementById("status").innerText =
                    result.status || "Transkribering avslutad. Startar nedladdning...";

                try {
                    await downloadResult(file_id, result.download_filename);
                    document.getElementById("status").innerText =
                        "Transkribering och analys avslutad. DOCX-fil laddas ner automatiskt.";
                } catch (e) {
                    console.error("Fel vid automatisk nedladdning:", e);
                    document.getElementById("status").innerText =
                        "Resultatet är klart, men den automatiska nedladdningen misslyckades. Ladda om sidan och försök igen innan serverfilen städas bort.";
                }
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
