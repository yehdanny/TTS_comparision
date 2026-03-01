const API_BASE = 'http://localhost:8000';
const MODELS = ['f5', 'cosyvoice', 'gptsovits'];

// ── Health check on load ──────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    for (const [model, info] of Object.entries(data.models)) {
      const key = model.replace(/-/g, '');  // "f5-tts" → "f5tts", keep as-is otherwise
      setStatus(model, info.loaded ? 'online' : 'offline');
    }
  } catch {
    MODELS.forEach(m => setStatus(m, 'offline'));
  }
}

function setStatus(model, state) {
  const el = document.getElementById(`status-${model}`);
  if (!el) return;
  el.className = `status-badge status-${state}`;
  el.textContent = state.charAt(0).toUpperCase() + state.slice(1);
}

// ── Read reference audio as base64 ───────────────────────────────────────────
async function getRefAudioB64() {
  const file = document.getElementById('ref-audio').files[0];
  if (!file) return null;
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // strip the data URL prefix, keep only base64
      const b64 = reader.result.split(',')[1];
      resolve(b64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// ── Generate for a single model ───────────────────────────────────────────────
async function generateModel(model) {
  const text = document.getElementById('tts-text').value.trim();
  if (!text) { alert('Please enter some text first.'); return; }

  setStatus(model, 'loading');
  showSpinner(model, true);
  hideAudio(model);

  let refAudio = null;
  try {
    refAudio = await getRefAudioB64();
  } catch {
    console.warn('Could not read reference audio file');
  }

  const body = { text };
  if (refAudio) body.reference_audio = refAudio;

  try {
    const res = await fetch(`${API_BASE}/api/tts/${model}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    displayResult(model, data);
    setStatus(model, 'online');
  } catch (err) {
    console.error(`[${model}] request failed:`, err);
    setStatus(model, 'offline');
    document.getElementById(`metric-${model}-time`).textContent = 'Error';
  } finally {
    showSpinner(model, false);
  }
}

// ── Generate all models in parallel ──────────────────────────────────────────
async function generateAll() {
  const btn = document.getElementById('btn-generate-all');
  btn.disabled = true;
  btn.textContent = 'Generating…';

  await Promise.allSettled(MODELS.map(m => generateModel(m)));

  btn.disabled = false;
  btn.textContent = 'Generate All';
}

// ── Display result in card ────────────────────────────────────────────────────
function displayResult(model, data) {
  const audioUrl = `${API_BASE}${data.audio_url}`;

  const audioEl = document.getElementById(`audio-${model}`);
  audioEl.src = audioUrl;

  const dlEl = document.getElementById(`download-${model}`);
  dlEl.href = audioUrl;
  dlEl.download = `${model}_output.wav`;

  document.getElementById(`audio-section-${model}`).style.display = 'flex';

  document.getElementById(`metric-${model}-time`).textContent =
    data.inference_time != null ? `${data.inference_time.toFixed(2)} s` : '—';
  document.getElementById(`metric-${model}-rtf`).textContent =
    data.rtf != null ? data.rtf.toFixed(3) : '—';
  document.getElementById(`metric-${model}-size`).textContent =
    data.file_size_kb != null ? `${data.file_size_kb} KB` : '—';
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function showSpinner(model, show) {
  document.getElementById(`spinner-${model}`).style.display = show ? 'block' : 'none';
}

function hideAudio(model) {
  document.getElementById(`audio-section-${model}`).style.display = 'none';
}

// ── Init ──────────────────────────────────────────────────────────────────────
checkHealth();
