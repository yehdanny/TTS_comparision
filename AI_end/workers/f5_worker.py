"""
F5-TTS persistent worker — runs in ttsenv.
Protocol: JSON-lines on stdin/stdout.
  stdin:  {"id":"...", "text":"...", "ref_audio":"/path or null", "ref_text":"..."}
  stdout: {"id":"...", "audio_path":"...", "inference_time":0.0, "rtf":0.0}
          {"id":"...", "error":"message"}
"""
import json
import os
import sys
import time
import uuid

# ── locate F5-TTS built-in reference files ───────────────────────────────────
import importlib.util
_spec   = importlib.util.find_spec("f5_tts")
_F5_PKG = list(_spec.submodule_search_locations)[0]
DEFAULT_REF_ZH      = os.path.join(_F5_PKG, "infer", "examples", "basic", "basic_ref_zh.wav")
DEFAULT_REF_EN      = os.path.join(_F5_PKG, "infer", "examples", "basic", "basic_ref_en.wav")
DEFAULT_REF_ZH_TEXT = "对，这就是我，万人敬仰的太乙真人。"
DEFAULT_REF_EN_TEXT = "Some call me nature, others call me mother nature."

# ── user-supplied default reference (ref_audio/reference.wav + .txt) ─────────
USER_REF_AUDIO     = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref_audio", "reference.wav")
)
USER_REF_TEXT_FILE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref_audio", "reference.txt")
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import soundfile as sf   # noqa: E402
import numpy as np       # noqa: E402


def _to_pcm_wav(src: str) -> str:
    """Re-encode audio as 16-bit PCM WAV for format normalisation before inference."""
    data, sr = sf.read(src, dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)   # mix to mono
    tmp = os.path.join(OUTPUT_DIR, f"tmpref_{uuid.uuid4().hex}.wav")
    sf.write(tmp, data, sr, subtype='PCM_16')
    return tmp


# ── load model ────────────────────────────────────────────────────────────────
print("LOADING", flush=True)
from f5_tts.api import F5TTS   # noqa: E402
tts = F5TTS()
print("READY", flush=True)


def _is_chinese(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def handle(req: dict) -> dict:
    req_id   = req.get("id", str(uuid.uuid4()))
    text     = req["text"]
    ref_path = req.get("ref_audio")

    # ── choose reference audio + transcript ───────────────────────────────────
    if ref_path and os.path.exists(ref_path):
        # caller-supplied reference (uploaded from frontend)
        ref_file = ref_path
        ref_text = req.get("ref_text", "")
    elif os.path.exists(USER_REF_AUDIO):
        # default user reference: ref_audio/reference.wav + reference.txt
        ref_file = USER_REF_AUDIO
        ref_text = ""
        if os.path.exists(USER_REF_TEXT_FILE):
            with open(USER_REF_TEXT_FILE, encoding="utf-8") as f:
                ref_text = f.read().strip()
    elif _is_chinese(text):
        # F5-TTS built-in Mandarin reference
        ref_file = DEFAULT_REF_ZH
        ref_text = DEFAULT_REF_ZH_TEXT
    else:
        # F5-TTS built-in English reference
        ref_file = DEFAULT_REF_EN
        ref_text = DEFAULT_REF_EN_TEXT

    out_path  = os.path.join(OUTPUT_DIR, f"f5_{uuid.uuid4().hex}.wav")
    start     = time.perf_counter()
    _tmp_ref  = _to_pcm_wav(ref_file)
    try:
        tts.infer(
            ref_file=_tmp_ref,
            ref_text=ref_text,
            gen_text=text,
            file_wave=out_path,
            remove_silence=False,
        )
    finally:
        if os.path.exists(_tmp_ref):
            os.remove(_tmp_ref)
    elapsed  = time.perf_counter() - start
    duration = sf.info(out_path).duration if os.path.exists(out_path) else 1.0

    return {
        "id":             req_id,
        "audio_path":     out_path,
        "inference_time": round(elapsed, 3),
        "rtf":            round(elapsed / max(duration, 0.01), 3),
    }


# ── main loop ─────────────────────────────────────────────────────────────────
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
