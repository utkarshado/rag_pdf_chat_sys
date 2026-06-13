const sessionKey = "rag_pdf_chat_session_id";

function getSessionId() {
  let sid = localStorage.getItem(sessionKey);
  if (!sid) {
    sid = crypto.randomUUID();
    localStorage.setItem(sessionKey, sid);
  }
  return sid;
}

const sessionId = getSessionId();

const pdfInput = document.getElementById("pdfInput");
const uploadBtn = document.getElementById("uploadBtn");
const resetBtn = document.getElementById("resetBtn");
const sendBtn = document.getElementById("sendBtn");
const questionInput = document.getElementById("questionInput");
const chatBox = document.getElementById("chatBox");
const currentDoc = document.getElementById("currentDoc");
const sessionLabel = document.getElementById("sessionLabel");

sessionLabel.textContent = `Session: ${sessionId.slice(0, 8)}`;

function scrollChatToBottom() {
  chatBox.scrollTop = chatBox.scrollHeight;
}

function addMessage(role, text, meta = "") {
  const wrapper = document.createElement("div");
  wrapper.className = role === "user" ? "message user-message" : "message bot-message";

  const metaEl = document.createElement("div");
  metaEl.className = "message-meta";
  metaEl.textContent = role === "user" ? "You" : "Assistant";

  const textEl = document.createElement("div");
  textEl.className = "message-text";
  textEl.textContent = text;

  wrapper.appendChild(metaEl);
  wrapper.appendChild(textEl);

  if (meta) {
    const metaInfo = document.createElement("div");
    metaInfo.className = "message-source";
    metaInfo.textContent = meta;
    wrapper.appendChild(metaInfo);
  }

  chatBox.appendChild(wrapper);
  scrollChatToBottom();
}

function setLoading(isLoading) {
  chatBox.classList.toggle("loading", isLoading);
  sendBtn.disabled = isLoading;
  uploadBtn.disabled = isLoading;
  resetBtn.disabled = isLoading;
}

async function uploadPdf() {
  const file = pdfInput.files[0];
  if (!file) {
    alert("Choose a PDF file first.");
    return;
  }

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("pdf", file);

  setLoading(true);
  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Upload failed");
    }

    currentDoc.textContent = data.pdf_name;
    addMessage("assistant", `PDF uploaded and indexed: ${data.pdf_name}`);
  } catch (err) {
    addMessage("assistant", `Upload error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  addMessage("user", question);
  questionInput.value = "";

  setLoading(true);
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: sessionId,
        question: question,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Chat failed");
    }

    const meta = data.sources && data.sources.length
      ? `Sources: ${data.sources.join(", ")}`
      : "";

    addMessage("assistant", data.answer, meta);
  } catch (err) {
    addMessage("assistant", `Error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

async function resetSession() {
  setLoading(true);
  try {
    const form = new FormData();
    form.append("session_id", sessionId);

    const res = await fetch("/api/reset", {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Reset failed");
    }

    currentDoc.textContent = "No PDF uploaded";
    chatBox.innerHTML = "";
    addMessage("assistant", "Session cleared. Upload a new PDF to continue.");
  } catch (err) {
    addMessage("assistant", `Reset error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

uploadBtn.addEventListener("click", uploadPdf);
sendBtn.addEventListener("click", sendQuestion);
resetBtn.addEventListener("click", resetSession);

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    sendQuestion();
  }
});

addMessage("assistant", "Upload a PDF to begin.");
