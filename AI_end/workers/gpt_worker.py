"""
GPT-SoVITS persistent worker — runs in venv_gpt.
Protocol: JSON-lines on stdin/stdout.
"""
import json
import os
import sys
import time
import uuid

# Add GPT-SoVITS repo to path
GPT_REPO = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "GPT-SoVITS")
)
sys.path.insert(0, GPT_REPO)

# Pretrained model paths (downloaded from HuggingFace lj1995/GPT-SoVITS)
PRETRAINED_DIR  = os.path.join(GPT_REPO, "GPT_SoVITS", "pretrained_models")
GPT_WEIGHTS     = os.path.join(PRETRAINED_DIR, "gsv-v2final-pretrained", "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
SOVITS_WEIGHTS  = os.path.join(PRETRAINED_DIR, "gsv-v2final-pretrained", "s2G2333k.pth")
BERT_PATH       = os.path.join(PRETRAINED_DIR, "chinese-roberta-wwm-ext-large")
CNHF_PATH       = os.path.join(PRETRAINED_DIR, "chinese-hubert-base")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("LOADING", flush=True)
os.chdir(GPT_REPO)  # GPT-SoVITS relies on relative paths internally

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

tts_config = TTS_Config(os.path.join(GPT_REPO, "GPT_SoVITS", "configs", "tts_infer.yaml"))
tts_config.device = "cuda"
tts_config.is_half = True
tts_config.t2s_weights_path   = GPT_WEIGHTS
tts_config.vits_weights_path  = SOVITS_WEIGHTS
tts_config.bert_base_path     = BERT_PATH
tts_config.cnhf_base_path     = CNHF_PATH

pipeline = TTS(tts_config)
print("READY", flush=True)


def handle(req: dict) -> dict:
    req_id   = req.get("id", str(uuid.uuid4()))
    text     = req["text"]
    ref_path = req.get("ref_audio")
    ref_text = req.get("ref_text", "")

    # GPT-SoVITS needs a reference audio; use a bundled default if none provided
    if not ref_path or not os.path.exists(ref_path):
        # Try to find any sample .wav shipped with the repo
        sample_dir = os.path.join(GPT_REPO, "tools", "asr", "demo_outputs")
        wavs = [f for f in os.listdir(sample_dir) if f.endswith(".wav")] if os.path.isdir(sample_dir) else []
        ref_path = os.path.join(sample_dir, wavs[0]) if wavs else None
        ref_text = ""

    if not ref_path:
        raise RuntimeError("GPT-SoVITS requires a reference audio file")

    out_path = os.path.join(OUTPUT_DIR, f"gptsovits_{uuid.uuid4().hex}.wav")
    start = time.perf_counter()

    # Detect language
    has_zh = any("\u4e00" <= c <= "\u9fff" for c in text)
    text_lang = "zh" if has_zh else "en"
    ref_lang  = "zh" if ref_text and any("\u4e00" <= c <= "\u9fff" for c in ref_text) else "en"

    inputs = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_path,
        "prompt_text": ref_text,
        "prompt_lang": ref_lang,
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "speed_factor": 1.0,
    }

    import soundfile as sf
    import numpy as np
    chunks = []
    sample_rate = 32000
    for sr, audio in pipeline.run(inputs):
        sample_rate = sr
        chunks.append(audio)

    audio_data = np.concatenate(chunks) if chunks else np.zeros(sample_rate, dtype=np.float32)
    sf.write(out_path, audio_data, sample_rate)

    elapsed  = time.perf_counter() - start
    duration = len(audio_data) / sample_rate

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
