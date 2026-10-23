// ✅ On page load: enable/disable fields based on API key presence
let globalEncryptionKeyBase64 = "";

// Accepted audio formats and the largest upload, replaced by the values from
// /config when it answers. The server checks both again.
let supportedAudioExtensions = [".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"];
let maxUploadMb = 1024;

// --- Editable instructions ---------------------------------------------------
// The standard texts are fetched from the backend so that
// policy/prompt_policy.json stays the single source of truth. An edited text
// is remembered between visits, in this browser only.

const SUMMARY_PROMPT_STORAGE_KEY = "jbg_summary_prompt";
const SUMMARY_STYLE_STORAGE_KEY = "jbg_summary_style";
const QUESTIONS_PROMPT_STORAGE_KEY = "jbg_questions_prompt";
const SPEAKERS_PROMPT_STORAGE_KEY = "jbg_speakers_prompt";
const CUSTOM_PROMPT_STORAGE_KEY = "jbg_custom_prompt";

// Used only if /analysis_prompts cannot be reached; the server always puts its
// own copy of the opening first.
const FALLBACK_CUSTOM_PREAMBLE =
    "Du analyserar och sammanställer transkriberad text från en ljudinspelning. Gör följande analys:";

// Option ids retired in an earlier version. Browsers still hold these, so a
// stored value is mapped rather than silently falling back to the default.
const LEGACY_SUMMARY_STYLE_IDS = { short: "enkel", extensive: "utforlig" };

let summaryOptions = [];              // [{id, label, description, default, prompt}]
let summaryPromptMaxLength = 20000;
let summaryOptionsLoaded = false;
let summaryPromptText = "";           // the summary instruction in use

// Standard texts for follow-up questions and speakers, and the fixed opening
// of the custom analysis. Null until /analysis_prompts has answered.
let analysisDefaults = null;
// What the user has written. Empty means the standard text; for the custom
// analysis it means that nothing has been written.
let questionsPromptText = "";
let speakersPromptText = "";
let customPromptText = "";

function readStored(key) {
    try {
        return localStorage.getItem(key) || "";
    } catch (err) {
        return "";
    }
}

function writeStored(key, value) {
    try {
        if (value) {
            localStorage.setItem(key, value);
        } else {
            localStorage.removeItem(key);
        }
    } catch (err) {
        console.warn("Kunde inte spara instruktionen lokalt:", err);
    }
}

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
    return summaryPromptText.trim() !== defaultSummaryPromptFor(currentSummaryStyle()).trim();
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

function showPromptState(id, edited) {
    // Nothing is shown while the standard text is in place - that is the
    // normal case. An edited instruction is called out, since it is the
    // state the user could otherwise forget they are in.
    const state = document.getElementById(id);
    if (!state) return;
    state.textContent = edited ? "Egen text används." : "";
    state.hidden = !edited;
}

function refreshSummaryPromptState() {
    if (!summaryOptionsLoaded) {
        const state = document.getElementById("summaryPromptState");
        state.textContent = "Standardtexterna kunde inte hämtas. Skriv en egen instruktion, annars används serverns standardtext.";
        state.hidden = false;
        return;
    }
    showPromptState("summaryPromptState", isSummaryPromptEdited());
}

function refreshAnalysisPromptStates() {
    showPromptState("questionsPromptState", questionsPromptText !== "");
    showPromptState("speakersPromptState", speakersPromptText !== "");
}

function applySummaryDefault(id) {
    summaryPromptText = defaultSummaryPromptFor(id);
    storeSummaryPrompt();
    refreshSummaryStyleDescription();
    refreshSummaryPromptState();
}

function storeSummaryPrompt() {
    try {
        localStorage.setItem(SUMMARY_PROMPT_STORAGE_KEY, summaryPromptText);
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

// An edited text equal to the standard text is no edit at all.
function editedOrEmpty(text, standard) {
    const trimmed = (text || "").trim();
    return trimmed === (standard || "").trim() ? "" : trimmed;
}

// --- What the editor can edit -------------------------------------------------
// One dialog serves every instruction. Each target says what to show, what
// its standard text is, and what to do on Klar and on Avbryt.

const editorTargets = {
    summary: {
        title: "Redigera promptinstruktioner",
        subtitle: () => `Utgår från: ${summaryStyleLabel(currentSummaryStyle())}`,
        prefix: () => "",
        text: () => summaryPromptText,
        standard: () => defaultSummaryPromptFor(currentSummaryStyle()),
        resetLabel: "Återgå till standardtexten",
        trigger: () => document.getElementById("openSummaryEditor"),
        apply: text => {
            summaryPromptText = text;
            storeSummaryPrompt();
            refreshSummaryPromptState();
        },
    },
    questions: {
        title: "Redigera instruktionen för följdfrågor",
        subtitle: () => "",
        prefix: () => "",
        text: () => questionsPromptText || (analysisDefaults ? analysisDefaults.follow_up_questions : ""),
        standard: () => (analysisDefaults ? analysisDefaults.follow_up_questions : ""),
        resetLabel: "Återgå till standardtexten",
        trigger: () => document.getElementById("openQuestionsEditor"),
        apply: text => {
            questionsPromptText = editedOrEmpty(text, editorTargets.questions.standard());
            writeStored(QUESTIONS_PROMPT_STORAGE_KEY, questionsPromptText);
            refreshAnalysisPromptStates();
        },
    },
    speakers: {
        title: "Redigera instruktionen för talaridentifiering",
        subtitle: () => "Beskriv inspelningen och vilka beteckningar talarna ska få. Formen på svaret, en replik per rad med talarens beteckning, läggs till automatiskt.",
        prefix: () => "",
        text: () => speakersPromptText || (analysisDefaults ? analysisDefaults.speaker_diarization : ""),
        standard: () => (analysisDefaults ? analysisDefaults.speaker_diarization : ""),
        resetLabel: "Återgå till standardtexten",
        trigger: () => document.getElementById("openSpeakersEditor"),
        apply: text => {
            speakersPromptText = editedOrEmpty(text, editorTargets.speakers.standard());
            writeStored(SPEAKERS_PROMPT_STORAGE_KEY, speakersPromptText);
            refreshAnalysisPromptStates();
        },
    },
    custom: {
        title: "Anpassad analys",
        subtitle: () => "Skriv vilken analys som ska göras. Den inledande meningen är fast och skickas alltid först.",
        prefix: () => (analysisDefaults && analysisDefaults.custom_analysis_preamble) || FALLBACK_CUSTOM_PREAMBLE,
        text: () => customPromptText,
        standard: () => "",
        resetLabel: "Rensa texten",
        trigger: () => document.getElementById("optCustom"),
        // Checked only if something has been written after the fixed opening.
        apply: text => {
            customPromptText = (text || "").trim();
            writeStored(CUSTOM_PROMPT_STORAGE_KEY, customPromptText);
            document.getElementById("optCustom").checked = customPromptText !== "";
        },
        cancel: () => {
            document.getElementById("optCustom").checked = false;
        },
    },
};

// --- Saved instructions -------------------------------------------------------
// A user can save an edited text under a name and pick it again later. Each
// kind of instruction has its own list, kept in this browser only, so nothing
// is stored on the server. Picking a saved text only fills in the field; Klar
// applies it as usual.

const SAVED_PROMPTS_KEY_PREFIX = "jbg_saved_prompts_";

// Name of the saved text last picked, offered as the name when saving again.
// Kept apart from the menu, which goes back to its placeholder as soon as the
// field no longer holds that text.
let lastPickedSavedName = "";
const MAX_SAVED_PROMPTS = 20;
const MAX_SAVED_PROMPT_NAME = 60;

function savedPromptsFor(key) {
    try {
        const list = JSON.parse(localStorage.getItem(SAVED_PROMPTS_KEY_PREFIX + key) || "[]");
        return Array.isArray(list)
            ? list.filter(e => e && typeof e.name === "string" && typeof e.text === "string")
            : [];
    } catch (err) {
        return [];
    }
}

function storeSavedPrompts(key, list) {
    try {
        if (list.length) {
            localStorage.setItem(SAVED_PROMPTS_KEY_PREFIX + key, JSON.stringify(list));
        } else {
            localStorage.removeItem(SAVED_PROMPTS_KEY_PREFIX + key);
        }
        return true;
    } catch (err) {
        console.warn("Kunde inte spara instruktionen lokalt:", err);
        return false;
    }
}

function refreshSavedPrompts(selectedName = "") {
    const key = activeEditorKey;
    const area = document.getElementById("promptEditorSaved");
    const select = document.getElementById("savedPromptSelect");
    const list = key ? savedPromptsFor(key) : [];

    area.hidden = list.length === 0;
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = `Sparade instruktioner (${list.length})`;
    select.replaceChildren(placeholder, ...list.map(entry => {
        const option = document.createElement("option");
        option.value = entry.name;
        option.textContent = entry.name;
        return option;
    }));
    select.value = list.some(e => e.name === selectedName) ? selectedName : "";
    document.getElementById("deleteSavedPrompt").hidden = select.value === "";
}

function saveCurrentPromptAs() {
    const key = activeEditorKey;
    const target = activeEditorTarget;
    if (!key || !target) return;
    const text = document.getElementById("promptEditorText").value.trim();

    if (!text || text === target.standard().trim()) {
        alert("Det finns ingen egen text att spara. Skriv eller ändra texten först.");
        return;
    }

    const answer = prompt("Ge instruktionen ett namn:", lastPickedSavedName);
    const name = (answer || "").trim().slice(0, MAX_SAVED_PROMPT_NAME);
    if (!name) return;

    const list = savedPromptsFor(key);
    const existing = list.find(e => e.name === name);
    if (existing) {
        if (!confirm(`Det finns redan en sparad instruktion som heter ”${name}”. Vill du ersätta den?`)) return;
        existing.text = text;
    } else {
        if (list.length >= MAX_SAVED_PROMPTS) {
            alert(`Du kan spara högst ${MAX_SAVED_PROMPTS} instruktioner av det här slaget. Ta bort någon först.`);
            return;
        }
        list.push({ name, text });
        list.sort((a, b) => a.name.localeCompare(b.name, "sv"));
    }
    if (storeSavedPrompts(key, list)) {
        lastPickedSavedName = name;
        refreshSavedPrompts(name);
    }
}

// The menu shows a name only while the field holds exactly that saved text.
// Once the text is reset, cleared or edited, the menu goes back to its
// placeholder, so the same name can be picked again in one step - picking
// the option already shown would not fire a change.
function releaseSavedPromptSelection() {
    const select = document.getElementById("savedPromptSelect");
    if (!select.value || !activeEditorKey) return;
    const entry = savedPromptsFor(activeEditorKey).find(e => e.name === select.value);
    const text = document.getElementById("promptEditorText").value;
    if (!entry || entry.text !== text) {
        select.value = "";
        document.getElementById("deleteSavedPrompt").hidden = true;
    }
}

function loadSelectedSavedPrompt() {
    const name = document.getElementById("savedPromptSelect").value;
    document.getElementById("deleteSavedPrompt").hidden = name === "";
    if (!name) return;
    const entry = savedPromptsFor(activeEditorKey).find(e => e.name === name);
    if (!entry) return;
    const textarea = document.getElementById("promptEditorText");
    textarea.value = entry.text;
    lastPickedSavedName = entry.name;
    refreshEditorReset();
    textarea.focus();
}

function deleteSelectedSavedPrompt() {
    const name = document.getElementById("savedPromptSelect").value;
    if (!name || !confirm(`Ta bort den sparade instruktionen ”${name}”? Texten i fältet ändras inte.`)) return;
    storeSavedPrompts(activeEditorKey, savedPromptsFor(activeEditorKey).filter(e => e.name !== name));
    refreshSavedPrompts();
}

// --- Floating editor ---------------------------------------------------------
// The instruction is edited in a <dialog> so that showing it never changes the
// height of the page. Native showModal() is used for the focus trap, the
// backdrop and Esc; only its default centring is overridden, because that
// centres on the iframe's own viewport rather than what the user can see.

let activeEditorTarget = null;      // the entry of editorTargets being edited
let activeEditorKey = null;         // its key: summary, questions, speakers, custom
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

function positionPromptEditor(trigger) {
    // Called after the dialog is shown, so its real height can be measured.
    const dialog = document.getElementById("promptEditor");
    const dialogHeight = dialog.offsetHeight || 460;
    const info = latestParentPageInfo;

    // Fallback: anchor above the control the user just clicked. It is on
    // screen by definition, so this is safe when nothing else is known.
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

function refreshEditorReset() {
    const target = activeEditorTarget;
    const reset = document.getElementById("resetPromptEditor");
    if (!target) return;
    const textarea = document.getElementById("promptEditorText");
    reset.textContent = target.resetLabel;
    reset.hidden = textarea.value.trim() === target.standard().trim();
}

function openPromptEditor(key) {
    const target = editorTargets[key];
    const dialog = document.getElementById("promptEditor");
    const textarea = document.getElementById("promptEditorText");

    activeEditorTarget = target;
    activeEditorKey = key;
    lastPickedSavedName = "";
    document.getElementById("promptEditorTitle").textContent = target.title;

    const subtitle = document.getElementById("promptEditorSubtitle");
    subtitle.textContent = target.subtitle();
    subtitle.hidden = subtitle.textContent === "";

    const prefix = document.getElementById("promptEditorPrefix");
    prefix.textContent = target.prefix();
    prefix.hidden = prefix.textContent === "";

    textarea.maxLength = summaryPromptMaxLength;
    textarea.value = target.text();
    refreshEditorReset();
    refreshSavedPrompts();

    if (typeof dialog.showModal === "function") {
        dialog.showModal();
    } else {
        dialog.setAttribute("open", "");   // very old browsers: no modality
    }
    positionPromptEditor(target.trigger());
    textarea.focus();
    textarea.setSelectionRange(0, 0);
    textarea.scrollTop = 0;
}

function closePromptEditor(apply) {
    const dialog = document.getElementById("promptEditor");
    const textarea = document.getElementById("promptEditorText");
    const target = activeEditorTarget;
    activeEditorTarget = null;
    activeEditorKey = null;

    if (target) {
        if (apply) {
            target.apply(textarea.value);
        } else if (target.cancel) {
            target.cancel();             // Avbryt, Esc and the backdrop
        }
    }

    if (dialog.open && typeof dialog.close === "function") {
        dialog.close();
    } else {
        dialog.removeAttribute("open");
    }
}

function setUpPromptEditorDialog() {
    const dialog = document.getElementById("promptEditor");
    const textarea = document.getElementById("promptEditorText");

    document.getElementById("openSummaryEditor")
        .addEventListener("click", () => openPromptEditor("summary"));
    document.getElementById("openQuestionsEditor")
        .addEventListener("click", () => openPromptEditor("questions"));
    document.getElementById("openSpeakersEditor")
        .addEventListener("click", () => openPromptEditor("speakers"));
    document.getElementById("applyPromptEditor")
        .addEventListener("click", () => closePromptEditor(true));
    document.getElementById("cancelPromptEditor")
        .addEventListener("click", () => closePromptEditor(false));

    document.getElementById("savePromptAs").addEventListener("click", saveCurrentPromptAs);
    document.getElementById("savedPromptSelect").addEventListener("change", loadSelectedSavedPrompt);
    document.getElementById("deleteSavedPrompt").addEventListener("click", deleteSelectedSavedPrompt);

    document.getElementById("resetPromptEditor").addEventListener("click", () => {
        if (!activeEditorTarget) return;
        textarea.value = activeEditorTarget.standard();
        refreshEditorReset();
        releaseSavedPromptSelection();
        textarea.focus();
    });

    textarea.addEventListener("input", () => {
        if (textarea.value.length > summaryPromptMaxLength) {
            textarea.value = textarea.value.slice(0, summaryPromptMaxLength);
        }
        refreshEditorReset();
        releaseSavedPromptSelection();
    });

    // Esc means the same here as everywhere else: discard and close.
    dialog.addEventListener("cancel", event => {
        event.preventDefault();
        closePromptEditor(false);
    });

    // Clicking the backdrop is treated as Avbryt.
    dialog.addEventListener("click", event => {
        if (event.target === dialog) closePromptEditor(false);
    });

    trackParentPageInfo();
}

// The custom analysis has no Redigera link: checking the box always opens the
// editor, and unchecking it keeps the text for next time.
function setUpCustomAnalysis() {
    const checkbox = document.getElementById("optCustom");
    checkbox.addEventListener("change", () => {
        if (checkbox.checked) {
            checkbox.checked = false;      // until Klar decides
            openPromptEditor("custom");
        }
    });
}

function setUpAnalysisPrompts() {
    customPromptText = readStored(CUSTOM_PROMPT_STORAGE_KEY).trim();

    fetch("/analysis_prompts")
        .then(res => res.json())
        .then(data => {
            analysisDefaults = data;
            if (data.max_length) summaryPromptMaxLength = data.max_length;
            // A stored text equal to the standard text is not an edit.
            questionsPromptText = editedOrEmpty(readStored(QUESTIONS_PROMPT_STORAGE_KEY), data.follow_up_questions);
            speakersPromptText = editedOrEmpty(readStored(SPEAKERS_PROMPT_STORAGE_KEY), data.speaker_diarization);
            refreshAnalysisPromptStates();
        })
        .catch(err => {
            console.warn("Kunde inte hämta standardtexter för analyserna:", err);
            questionsPromptText = readStored(QUESTIONS_PROMPT_STORAGE_KEY).trim();
            speakersPromptText = readStored(SPEAKERS_PROMPT_STORAGE_KEY).trim();
            refreshAnalysisPromptStates();
        });
}

function setUpSummaryPromptEditor() {
    const select = document.getElementById("summaryStyle");

    setUpPromptEditorDialog();
    setUpCustomAnalysis();
    setUpAnalysisPrompts();

    // Remembered so the selection can be restored if the user cancels.
    let previousStyle = null;

    select.addEventListener("focus", () => { previousStyle = select.value; });

    select.addEventListener("change", () => {
        const newStyle = select.value;
        const edited = !matchesAnySummaryDefault(summaryPromptText) && summaryPromptText.trim() !== "";

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

            summaryPromptText = hasCustomText ? savedPrompt : defaultSummaryPromptFor(select.value);

            refreshSummaryStyleDescription();
            refreshSummaryPromptState();
        })
        .catch(err => {
            console.warn("Kunde inte hämta standardtexter för sammanfattning:", err);
            summaryOptionsLoaded = false;
            refreshSummaryPromptState();
        });
}

// Version, commit och miljö under rubriken, t.ex. "Version 1.1.0 · a1b2c3d (dev)".
// Commit länkar till GitHub när adressen är känd. Texten sätts som ren text.
function showVersion(version) {
  const line = document.getElementById("versionDisplay");
  if (!line || !version || !version.version) return;
  line.textContent = `Version ${version.version}`;
  if (version.commit_short) {
    line.append(" · ");
    if (version.commit_url) {
      const link = document.createElement("a");
      link.href = version.commit_url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = version.commit_short;
      link.title = `Commit ${version.commit}${version.branch ? " på grenen " + version.branch : ""}`;
      line.append(link);
    } else {
      line.append(version.commit_short);
    }
  }
  line.append(` (${version.environment})`);
  line.hidden = false;
}

// The model menu comes from policy/model_prices.json through /config, so a
// model can only be offered if it has a price. Each option shows the input
// and output price per million tokens, with the currency in the label to
// keep the options short on a phone; the tooltip adds the cached price.
const priceFormat = new Intl.NumberFormat("sv-SE", { maximumFractionDigits: 4 });

function fillModelMenu(models, defaultModel, prices = {}, currency = "USD") {
  const select = document.getElementById("modelSelect");
  if (!select || !Array.isArray(models) || !models.length) return;
  const currencyLabel = document.getElementById("modelPriceCurrency");
  if (currencyLabel) currencyLabel.textContent = currency;
  select.replaceChildren(...models.map(name => {
    const option = document.createElement("option");
    const price = prices && prices[name];
    option.value = name;
    option.textContent = price
      ? `${name} (indata: ${priceFormat.format(price.in)}, utdata: ${priceFormat.format(price.ut)})`
      : name;
    if (price) {
      option.title = `Per miljon tokens: indata ${priceFormat.format(price.in)} ${currency}, ` +
        `cachad indata ${priceFormat.format(price.cachad)} ${currency}, ` +
        `utdata ${priceFormat.format(price.ut)} ${currency}`;
    }
    option.selected = name === defaultModel;
    return option;
  }));
}

fetch("/config")
  .then(res => res.json())
  .then(data => {
    document.title = data.title;
    document.getElementById("app-title").innerText = data.title;
    showVersion(data.version);
    if (Array.isArray(data.audio_extensions) && data.audio_extensions.length) {
      supportedAudioExtensions = data.audio_extensions;
      document.getElementById("audioFile").accept = supportedAudioExtensions.join(",");
    }
    if (data.max_upload_mb > 0) {
      maxUploadMb = data.max_upload_mb;
    }
    fillModelMenu(data.models, data.default_model, data.model_prices, data.currency);
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

    const accuracySelect = document.getElementById("accuracySelect");
    const savedAccuracy = localStorage.getItem("jbg_transcription_accuracy");
    if (savedAccuracy && accuracySelect.querySelector(`option[value="${savedAccuracy}"]`)) {
        accuracySelect.value = savedAccuracy;
    }
    accuracySelect.addEventListener("change", () => {
        try {
            localStorage.setItem("jbg_transcription_accuracy", accuracySelect.value);
        } catch (err) {
            console.warn("Kunde inte spara vald noggrannhet lokalt:", err);
        }
    });

    const apiKeyInput = document.getElementById("apiKey");
    const checkboxes = [
        document.getElementById("optSummary"),
        document.getElementById("optQuestions"),
        document.getElementById("optSpeakers"),
        document.getElementById("optCustom"),
    ];

    // Marking suspected errors is done by the language model, so that
    // presentation needs an API key. Raw text and timestamps do not.
    const markedFormatRadio = document.querySelector('input[name="transcriptionFormat"][value="marked"]');

    apiKeyInput.addEventListener("input", () => {
        const hasKey = apiKeyInput.value.trim().length > 0;
        checkboxes.forEach(cb => cb.disabled = !hasKey);
        markedFormatRadio.disabled = !hasKey;
        if (!hasKey && markedFormatRadio.checked) {
            document.querySelector('input[name="transcriptionFormat"][value="raw"]').checked = true;
        }
    });

    markedFormatRadio.disabled = apiKeyInput.value.trim().length === 0;

    // Summary checkbox: reveal the style choice and the editable instruction
    const summaryCheckbox = document.getElementById("optSummary");
    const summaryOptions = document.getElementById("summaryOptions");

    summaryCheckbox.addEventListener("change", () => {
        const show = summaryCheckbox.checked;
        summaryOptions.hidden = !show;
        document.getElementById("summaryStyle").disabled = !show;
        document.getElementById("openSummaryEditor").disabled = !show;
    });

    // Follow-up questions and speakers: reveal their Redigera link when checked.
    [["optQuestions", "questionsOptions", "openQuestionsEditor"],
     ["optSpeakers", "speakersOptions", "openSpeakersEditor"]].forEach(([boxId, blockId, linkId]) => {
        const box = document.getElementById(boxId);
        box.addEventListener("change", () => {
            document.getElementById(blockId).hidden = !box.checked;
            document.getElementById(linkId).disabled = !box.checked;
        });
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
    // By extension rather than by file.type: browsers report m4a and wav under
    // several different MIME types, and sometimes none at all.
    const dot = file.name.lastIndexOf(".");
    const extension = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
    if (!supportedAudioExtensions.includes(extension)) {
        const allowed = supportedAudioExtensions.map(e => e.slice(1)).join(", ");
        alert(`Filformatet stöds inte. Tillåtna format: ${allowed}.`);
        return;
    }
    if (file.size > maxUploadMb * 1024 * 1024) {
        const sizeMb = Math.ceil(file.size / (1024 * 1024));
        alert(`Filen är ${sizeMb} MB. Den största tillåtna storleken är ${maxUploadMb} MB.`);
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
    document.querySelectorAll('input[name="transcriptionFormat"]').forEach(rb => rb.disabled = true);
    document.getElementById("accuracySelect").disabled = true;
    document.getElementById("fastEngine").disabled = true;
    document.getElementById("optQuestions").disabled = true;
    document.getElementById("openQuestionsEditor").disabled = true;
    document.getElementById("optSpeakers").disabled = true;
    document.getElementById("openSpeakersEditor").disabled = true;
    document.getElementById("optCustom").disabled = true;
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
        summarizeChecked ? summaryPromptText.trim() : ""
    );
    formData.append("transcription_accuracy", document.getElementById("accuracySelect").value);
    formData.append("fast_engine", document.getElementById("fastEngine").checked);
    formData.append(
        "transcription_format",
        document.querySelector('input[name="transcriptionFormat"]:checked')?.value || "raw"
    );
    const questionsChecked = document.getElementById("optQuestions").checked;
    const speakersChecked = document.getElementById("optSpeakers").checked;
    const customChecked = document.getElementById("optCustom").checked;
    formData.append("questions", questionsChecked);
    formData.append("speakers", speakersChecked);
    // Empty means the standard text; only an edited text is sent.
    formData.append("questions_prompt", questionsChecked ? questionsPromptText : "");
    formData.append("speakers_prompt", speakersChecked ? speakersPromptText : "");
    // Only what follows the fixed opening; the server adds the opening.
    formData.append("custom_analysis", customChecked);
    formData.append("custom_prompt", customChecked ? customPromptText : "");

    fileInput.disabled = true;

    // Check Formdata for errors.
    //
    // Fälten loggas för felsökning, men innehållet får inte skrivas ut rakt av:
    // formuläret innehåller API-nyckeln och krypteringsnyckeln, och
    // webbläsarkonsolen sparas, klistras in i felrapporter och syns för den som
    // sitter bredvid. Hemligheter visas därför bara med längd, och
    // promptinstruktionen bara med teckenantal.
    const SECRET_FIELDS = ["api_key", "encryption_key"];
    const BULKY_FIELDS = ["summary_prompt"];
    for (const [key, value] of formData.entries()) {
        if (value instanceof Blob) {
            console.log(`${key}: [Blob] filename=${value.name}, type=${value.type}, size=${value.size}`);
        } else if (SECRET_FIELDS.includes(key)) {
            console.log(`${key}: [dolt, ${String(value).length} tecken]`);
        } else if (BULKY_FIELDS.includes(key)) {
            console.log(`${key}: [${String(value).length} tecken]`);
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

// Turns a failed download into something that can be acted on without opening
// the developer console, which is the only place the status was visible before.
function describeDownloadFailure(error) {
    const status = error && error.status;
    if (!status) {
        // No response at all: the request never completed.
        return " (ingen kontakt med servern).";
    }
    const reasons = {
        400: "nyckeln för kryptering saknades",
        404: "resultatfilen hittades inte på servern",
        500: "servern kunde inte läsa resultatfilen",
        502: "servern svarade inte i tid",
        503: "tjänsten var inte tillgänglig",
        504: "servern svarade inte i tid",
    };
    const reason = reasons[status];
    return reason ? ` (${status}: ${reason}).` : ` (statuskod ${status}).`;
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
        const error = new Error(`Nedladdningen misslyckades (${response.status}): ${text}`);
        // Kept separately so the status can be shown in the GUI without the
        // body, which may contain details that do not belong on screen.
        error.status = response.status;
        throw error;
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
                        "Resultatet är klart, men den automatiska nedladdningen misslyckades"
                        + describeDownloadFailure(e)
                        + " Ladda om sidan och försök igen innan serverfilen städas bort.";
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
