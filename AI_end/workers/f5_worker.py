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

# ── inject imageio-ffmpeg binary into PATH so torchaudio/audioread can find it ─
try:
    import imageio_ffmpeg as _iff
    _ffmpeg_dir = os.path.dirname(_iff.get_ffmpeg_exe())
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    del _iff, _ffmpeg_dir
except ImportError:
    pass  # install with: pip install imageio-ffmpeg  (inside ttsenv)

# ── locate built-in reference files ──────────────────────────────────────────
import importlib.util
_spec = importlib.util.find_spec("f5_tts")
_F5_PKG = list(_spec.submodule_search_locations)[0]
DEFAULT_REF_AUDIO = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref_audio", "reference.wav")
)
DEFAULT_REF_TEXT_FILE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref_audio", "reference.txt")
)
if DEFAULT_REF_AUDIO is None or not os.path.exists(DEFAULT_REF_AUDIO):
    DEFAULT_REF_AUDIO = os.path.join(_F5_PKG, "infer", "examples", "basic", "basic_ref_en.wav")
    DEFAULT_REF_TEXT_FILE = "Some call me nature, others call me mother nature."

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import soundfile as sf   # noqa: E402
import numpy as np       # noqa: E402


def _to_pcm_wav(src: str) -> str:
    """Re-encode audio as 16-bit PCM WAV using soundfile.
    Guarantees torchaudio/librosa can load it without ffmpeg."""
    data, sr = sf.read(src, dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # mix to mono
    tmp = os.path.join(OUTPUT_DIR, f"tmpref_{uuid.uuid4().hex}.wav")
    sf.write(tmp, data, sr, subtype='PCM_16')
    return tmp


# ── force torchaudio to use soundfile backend (no FFmpeg/DLLs required) ───────
# torchaudio >= 2.0 defaults to FFmpeg shared-library backend; on Windows without
# system FFmpeg that fails even for plain WAV files.  soundfile handles PCM WAV
# natively and is already installed in ttsenv.
import torchaudio as _ta
try:
    # torchaudio < 2.1 — set_audio_backend still exists
    _ta.set_audio_backend("soundfile")
except Exception:
    # torchaudio >= 2.1 removed set_audio_backend — monkey-patch load() directly
    def _sf_load(filepath, frame_offset=0, num_frames=-1, normalize=True,
                 channels_first=True, format=None, **kw):
        import torch
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True,
                           start=frame_offset,
                           frames=num_frames if num_frames > 0 else -1)
        tensor = torch.from_numpy(data.T if channels_first else data)
        return tensor, sr
    _ta.load = _sf_load
del _ta

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
    else:
        ref_file = DEFAULT_REF_AUDIO
        ref_text = DEFAULT_REF_TEXT_FILE

    out_path = os.path.join(OUTPUT_DIR, f"f5_{uuid.uuid4().hex}.wav")
    start = time.perf_counter()
    # Pre-encode as PCM-16 WAV so torchaudio/librosa can load without ffmpeg
    _tmp_ref = _to_pcm_wav(ref_file)
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
    elapsed = time.perf_counter() - start

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
