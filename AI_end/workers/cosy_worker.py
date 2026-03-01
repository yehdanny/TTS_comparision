"""
CosyVoice persistent worker — runs in venv_cosy.
Protocol: JSON-lines on stdin/stdout.
"""
import json
import os
import sys
import time
import uuid

# Add CosyVoice repo to path
COSY_REPO = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "CosyVoice")
)
sys.path.insert(0, COSY_REPO)
sys.path.insert(0, os.path.join(COSY_REPO, "third_party", "Matcha-TTS"))

MODEL_DIR = os.path.join(COSY_REPO, "pretrained_models", "CosyVoice2-0.5B")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("LOADING", flush=True)
from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402
import torchaudio                                # noqa: E402

model = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False)
SAMPLE_RATE = model.sample_rate

# Available SFT speakers for no-reference mode
_DEFAULT_SPEAKERS = model.list_available_spks() if hasattr(model, "list_available_spks") else []
_ZH_SPEAKER = next((s for s in _DEFAULT_SPEAKERS if "中" in s or "zh" in s.lower()), None) or (_DEFAULT_SPEAKERS[0] if _DEFAULT_SPEAKERS else None)

print("READY", flush=True)


def _collect_audio(generator) -> "torch.Tensor":
    import torch
    chunks = [chunk["tts_speech"] for chunk in generator]
    return torch.cat(chunks, dim=1)


def handle(req: dict) -> dict:
    req_id   = req.get("id", str(uuid.uuid4()))
    text     = req["text"]
    ref_path = req.get("ref_audio")
    ref_text = req.get("ref_text", "")

    out_path = os.path.join(OUTPUT_DIR, f"cosyvoice_{uuid.uuid4().hex}.wav")
    start = time.perf_counter()

    if ref_path and os.path.exists(ref_path):
        # Voice cloning (zero-shot)
        ref_audio, sr = torchaudio.load(ref_path)
        if sr != SAMPLE_RATE:
            ref_audio = torchaudio.functional.resample(ref_audio, sr, SAMPLE_RATE)
        audio = _collect_audio(
            model.inference_zero_shot(text, ref_text, ref_audio, stream=False)
        )
    elif _ZH_SPEAKER:
        # SFT (preset speaker)
        audio = _collect_audio(
            model.inference_sft(text, _ZH_SPEAKER, stream=False)
        )
    else:
        raise RuntimeError("No speakers available and no reference audio provided")

    torchaudio.save(out_path, audio, SAMPLE_RATE)
    elapsed = time.perf_counter() - start
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
