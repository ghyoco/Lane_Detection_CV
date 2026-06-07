// Point this at your deployed backend (e.g. "https://lane-api.onrender.com").
// Defaults to a local FastAPI dev server (`uvicorn main:app --reload`).
const API_URL = "http://localhost:8000";

const MAX_BYTES = 100 * 1024 * 1024;

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const statusEl = document.getElementById("status");
const videosEl = document.getElementById("videos");
const originalVideo = document.getElementById("originalVideo");
const resultVideo = document.getElementById("resultVideo");
const actionsEl = document.getElementById("actions");
const processBtn = document.getElementById("processBtn");
const resetBtn = document.getElementById("resetBtn");

let selectedFile = null;

function setStatus(message, kind) {
  statusEl.textContent = message || "";
  statusEl.hidden = !message;
  statusEl.className = "status" + (kind ? ` ${kind}` : "");
}

function formatBytes(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function reset() {
  selectedFile = null;
  fileInput.value = "";
  originalVideo.removeAttribute("src");
  resultVideo.removeAttribute("src");
  videosEl.hidden = true;
  actionsEl.hidden = true;
  dropzone.hidden = false;
  setStatus("");
}

function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith("video/")) {
    setStatus("Please choose a video file.", "error");
    return;
  }
  if (file.size > MAX_BYTES) {
    setStatus(`That file is ${formatBytes(file.size)} — the limit is 100 MB.`, "error");
    return;
  }

  selectedFile = file;
  originalVideo.src = URL.createObjectURL(file);
  resultVideo.removeAttribute("src");

  dropzone.hidden = true;
  videosEl.hidden = false;
  actionsEl.hidden = false;
  processBtn.disabled = false;
  processBtn.textContent = "Detect lanes";
  setStatus("");
}

async function processVideo() {
  if (!selectedFile) return;

  processBtn.disabled = true;
  resetBtn.disabled = true;
  setStatus("Uploading and processing — this can take a minute or two…", "busy");

  try {
    const formData = new FormData();
    formData.append("video", selectedFile);
    formData.append("mode", "auto");

    const res = await fetch(`${API_URL}/api/process`, { method: "POST", body: formData });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || `Processing failed (HTTP ${res.status}).`);
    }

    const blob = await res.blob();
    resultVideo.src = URL.createObjectURL(blob);
    setStatus("Done — lane overlay applied to the first ~10 seconds.");
  } catch (err) {
    setStatus(err.message || "Something went wrong. Is the backend running?", "error");
  } finally {
    processBtn.disabled = false;
    resetBtn.disabled = false;
  }
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => handleFile(e.target.files?.[0] ?? null));

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files?.[0] ?? null));

processBtn.addEventListener("click", processVideo);
resetBtn.addEventListener("click", reset);
