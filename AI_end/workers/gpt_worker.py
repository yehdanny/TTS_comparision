"""
GPT-SoVITS persistent worker — runs in venv_gpt.
Protocol: JSON-lines on stdin/stdout.
"""
import json
import os
import sys
import time
import uuid

# ── repo path: workers/ → AI_end/ → project_tts/ → claude_place/ → GPT-SoVITS/
GPT_REPO = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "GPT-SoVITS")
)
sys.path.insert(0, GPT_REPO)
# AR, module, text, etc. are inside GPT_SoVITS/ and use bare absolute imports
sys.path.insert(0, os.path.join(GPT_REPO, "GPT_SoVITS"))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_REF_AUDIO = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref_audio", "reference.wav")
)

# fast-langdetect expects its cache directory to exist before model load
_LANGDETECT_CACHE = os.path.join(GPT_REPO, "GPT_SoVITS", "pretrained_models", "fast_langdetect")
os.makedirs(_LANGDETECT_CACHE, exist_ok=True)

# NLTK data required by G2P / text processing
import nltk   # noqa: E402
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# GPT-SoVITS resolves all model paths relative to the repo root
os.chdir(GPT_REPO)

print("LOADING", flush=True)

from TTS_infer_pack.TTS import TTS, TTS_Config   # noqa: E402

# Load using tts_infer.yaml (custom section: cuda, is_half=True, v2 weights)
CONFIG_PATH = os.path.join("GPT_SoVITS", "configs", "tts_infer.yaml")
config = TTS_Config(CONFIG_PATH)
config.configs["device"]  = "cuda"
config.configs["is_half"] = True

pipeline = TTS(config)

print("READY", flush=True)


def _detect_lang(text: str) -> str:
    """Map text content to GPT-SoVITS language code."""
    has_zh = any("\u4e00" <= c <= "\u9fff" for c in text)
    has_en = any(c.isascii() and c.isalpha() for c in text)
    if has_zh and has_en:
        return "zh"       # mixed zh/en
    if has_zh:
        return "all_zh"   # pure Chinese
    return "en"           # pure English


def handle(req: dict) -> dict:
    req_id   = req.get("id", str(uuid.uuid4()))
    text     = req["text"]
    ref_path = req.get("ref_audio")
    ref_text = req.get("ref_text", "")

    if not ref_path or not os.path.exists(ref_path):
        if os.path.exists(DEFAULT_REF_AUDIO):
            ref_path = DEFAULT_REF_AUDIO
        else:
            raise RuntimeError(
                "GPT-SoVITS requires a reference audio file. "
                "Please upload a WAV file or place one at ref_audio/reference.wav."
            )

    out_path  = os.path.join(OUTPUT_DIR, f"gptsovits_{uuid.uuid4().hex}.wav")
    text_lang = _detect_lang(text)
    ref_lang  = _detect_lang(ref_text) if ref_text else text_lang

    inputs = {
        "text":             text,
        "text_lang":        text_lang,
        "ref_audio_path":   ref_path,
        "prompt_text":      ref_text,
        "prompt_lang":      ref_lang,
        "top_k":            15,
        "top_p":            1.0,
        "temperature":      1.0,
        "speed_factor":     1.0,
        "text_split_method": "cut1",
        "batch_size":       1,
    }

    import soundfile as sf
    import numpy as np

    start   = time.perf_counter()
    chunks  = []
    sr_out  = 32000
    for sr, audio in pipeline.run(inputs):
        sr_out = sr
        chunks.append(audio if hasattr(audio, "__len__") else audio)

    audio_data = np.concatenate(chunks) if chunks else np.zeros(sr_out, dtype=np.float32)
    sf.write(out_path, audio_data, sr_out)
    elapsed  = time.perf_counter() - start
    duration = len(audio_data) / sr_out

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
