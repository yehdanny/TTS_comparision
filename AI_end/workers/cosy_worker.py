"""
CosyVoice persistent worker — runs in venv_cosy.
Protocol: JSON-lines on stdin/stdout.
"""
import json
import os
import sys
import time
import uuid

# ── repo path: workers/ → AI_end/ → project_tts/ → claude_place/ → CosyVoice/
COSY_REPO = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "CosyVoice")
)
sys.path.insert(0, COSY_REPO)
sys.path.insert(0, os.path.join(COSY_REPO, "third_party", "Matcha-TTS"))

MODEL_DIR  = os.path.join(COSY_REPO, "pretrained_models", "CosyVoice2-0.5B")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_REF_AUDIO = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref_audio", "reference.wav")
)
DEFAULT_REF_TEXT_FILE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref_audio", "reference.txt")
)

print("LOADING", flush=True)

from cosyvoice.cli.cosyvoice import CosyVoice2   # noqa: E402
import torchaudio                                  # noqa: E402

# ── Traditional → Simplified Chinese converter ────────────────────────────────
# CosyVoice2's tokenizer/LangSegment is trained on Simplified Chinese.
# Traditional Chinese characters (語/语, 術/术, 來/来 …) are different code points
# and cause mis-tokenisation or mis-language-detection (often detected as Japanese).
try:
    from opencc import OpenCC as _OpenCC
    _t2s = _OpenCC('t2s')
    def _to_simplified(text: str) -> str:
        return _t2s.convert(text)
except ImportError:
    def _to_simplified(text: str) -> str:
        return text   # fallback: pass through (install opencc-python-reimplemented in venv_cosy)

model       = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False)
SAMPLE_RATE = model.sample_rate
_SPEAKERS   = model.list_available_spks()
# pick a Mandarin speaker for default (no reference audio) mode
_DEFAULT_SPK = next(
    (s for s in _SPEAKERS if "中" in s or "zh" in s.lower()),
    _SPEAKERS[0] if _SPEAKERS else None,
)

print("READY", flush=True)


def _collect_audio(generator):
    import torch
    chunks = [chunk["tts_speech"] for chunk in generator]
    return torch.cat(chunks, dim=1)


def handle(req: dict) -> dict:
    req_id   = req.get("id", str(uuid.uuid4()))
    text     = req["text"]
    ref_path = req.get("ref_audio")
    ref_text = req.get("ref_text", "")

    out_path = os.path.join(OUTPUT_DIR, f"cosyvoice_{uuid.uuid4().hex}.wav")
    start    = time.perf_counter()

    if (not ref_path or not os.path.exists(ref_path)) and os.path.exists(DEFAULT_REF_AUDIO):
        ref_path = DEFAULT_REF_AUDIO

    # Fall back to reference.txt for the prompt transcript
    if not ref_text and os.path.exists(DEFAULT_REF_TEXT_FILE):
        with open(DEFAULT_REF_TEXT_FILE, encoding="utf-8") as f:
            ref_text = f.read().strip()

    # Convert Traditional Chinese → Simplified so CosyVoice2 tokenizer/LangSegment works correctly
    text_simp     = _to_simplified(text)
    ref_text_simp = _to_simplified(ref_text)

    if ref_path and os.path.exists(ref_path):
        # inference_zero_shot internally calls load_wav(path, 16000) → torchaudio.load(path)
        # so we must pass the file path string, NOT a pre-loaded tensor
        audio = _collect_audio(
            model.inference_zero_shot(text_simp, ref_text_simp, ref_path, stream=False)
        )
    elif _DEFAULT_SPK:
        audio = _collect_audio(
            model.inference_sft(text_simp, _DEFAULT_SPK, stream=False)
        )
    else:
        raise RuntimeError("No speakers available and no reference audio provided")

    torchaudio.save(out_path, audio, SAMPLE_RATE)
    elapsed  = time.perf_counter() - start
    duration = audio.shape[-1] / SAMPLE_RATE

    return {
        "id": req_id,
        "audio_path": out_path,
        "inference_time": round(elapsed, 3),
        "rtf": round(elapsed / max(duration, 0.01), 3),
    }


for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        req  = json.loads(raw)
        resp = handle(req)
    except Exception as exc:
        resp = {"id": req.get("id", ""), "error": str(exc)}
    print(json.dumps(resp), flush=True)
