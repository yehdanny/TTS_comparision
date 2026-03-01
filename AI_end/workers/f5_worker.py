"""
F5-TTS persistent worker — runs in ttsenv.
Protocol: JSON-lines on stdin/stdout.
  stdin:  {"id":"...", "text":"...", "ref_audio":"/path or null"}
  stdout: {"id":"...", "audio_path":"...", "inference_time":0.0, "rtf":0.0}
          {"id":"...", "error":"message"}
"""
import json
import os
import sys
import time
import uuid

# ── locate built-in reference files ──────────────────────────────────────────
import importlib.util
_spec = importlib.util.find_spec("f5_tts")
_F5_PKG = list(_spec.submodule_search_locations)[0]
DEFAULT_REF_ZH = os.path.join(_F5_PKG, "infer", "examples", "basic", "basic_ref_zh.wav")
DEFAULT_REF_EN = os.path.join(_F5_PKG, "infer", "examples", "basic", "basic_ref_en.wav")
DEFAULT_REF_ZH_TEXT = "对，这就是我，万人敬仰的太乙真人。"
DEFAULT_REF_EN_TEXT = "Some call me nature, others call me mother nature."

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load model ────────────────────────────────────────────────────────────────
print("LOADING", flush=True)
from f5_tts.api import F5TTS   # noqa: E402  (import after path setup)
tts = F5TTS()
print("READY", flush=True)


def _is_chinese(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def handle(req: dict) -> dict:
    req_id   = req.get("id", str(uuid.uuid4()))
    text     = req["text"]
    ref_path = req.get("ref_audio")

    # Choose reference
    if ref_path and os.path.exists(ref_path):
        ref_file = ref_path
        ref_text = ""          # auto-transcribe
    elif _is_chinese(text):
        ref_file = DEFAULT_REF_ZH
        ref_text = DEFAULT_REF_ZH_TEXT
    else:
        ref_file = DEFAULT_REF_EN
        ref_text = DEFAULT_REF_EN_TEXT

    out_path = os.path.join(OUTPUT_DIR, f"f5_{uuid.uuid4().hex}.wav")
    start = time.perf_counter()
    tts.infer(
        ref_file=ref_file,
        ref_text=ref_text,
        gen_text=text,
        file_wave=out_path,
        remove_silence=True,
    )
    elapsed = time.perf_counter() - start

    import soundfile as sf
    duration = sf.info(out_path).duration if os.path.exists(out_path) else 1.0
    return {
        "id": req_id,
        "audio_path": out_path,
        "inference_time": round(elapsed, 3),
        "rtf": round(elapsed / max(duration, 0.01), 3),
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
