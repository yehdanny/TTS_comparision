"""
TTS Comparison — FastAPI backend
Run: python server.py
"""

import base64
import logging
import os
import tempfile
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import MODEL_PATHS, OUTPUT_DIR, PORT
from models.f5_tts import F5TTSModel
from models.cosyvoice import CosyVoiceModel
from models.gpt_sovits import GPTSoVITsModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model instances (loaded once at startup) ──────────────────────────────────
models = {
    "f5":         F5TTSModel(MODEL_PATHS["f5_tts"], OUTPUT_DIR),
    "cosyvoice":  CosyVoiceModel(MODEL_PATHS["cosyvoice"], OUTPUT_DIR),
    "gptsovits":  GPTSoVITsModel(MODEL_PATHS["gpt_sovits"], OUTPUT_DIR),
}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="TTS Comparison API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allows file:// and any localhost port
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    reference_audio: str | None = None   # base64-encoded audio (optional)


class TTSResponse(BaseModel):
    model: str
    audio_url: str
    inference_time: float
    rtf: float
    file_size_kb: float


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save_ref_audio(b64_data: str) -> str:
    """Decode base64 reference audio to a temp file; return the path."""
    raw = base64.b64decode(b64_data)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav", dir=OUTPUT_DIR, delete=False
    )
    tmp.write(raw)
    tmp.close()
    return tmp.name


def _build_response(model_name: str, result: dict) -> TTSResponse:
    audio_path = result["audio_path"]
    size_kb = round(os.path.getsize(audio_path) / 1024, 1)
    filename = Path(audio_path).name
    return TTSResponse(
        model=model_name,
        audio_url=f"/api/audio/{filename}",
        inference_time=round(result["inference_time"], 3),
        rtf=round(result["rtf"], 3),
        file_size_kb=size_kb,
    )


def _run_model(key: str, model_label: str, req: TTSRequest) -> TTSResponse:
    model = models[key]
    ref_path = None
    if req.reference_audio:
        try:
            ref_path = _save_ref_audio(req.reference_audio)
        except Exception as exc:
            logger.warning("Could not decode reference audio: %s", exc)
    try:
        result = model.generate(req.text, ref_path)
    except Exception as exc:
        logger.error("[%s] generation error: %s", model_label, exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
    return _build_response(model_label, result)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models": {
            "f5":        {"loaded": models["f5"].is_loaded()},
            "cosyvoice": {"loaded": models["cosyvoice"].is_loaded()},
            "gptsovits": {"loaded": models["gptsovits"].is_loaded()},
        },
    }


@app.post("/api/tts/f5", response_model=TTSResponse)
def tts_f5(req: TTSRequest):
    return _run_model("f5", "f5", req)


@app.post("/api/tts/cosyvoice", response_model=TTSResponse)
def tts_cosyvoice(req: TTSRequest):
    return _run_model("cosyvoice", "cosyvoice", req)


@app.post("/api/tts/gptsovits", response_model=TTSResponse)
def tts_gptsovits(req: TTSRequest):
    return _run_model("gptsovits", "gptsovits", req)


@app.get("/api/audio/{filename}")
def serve_audio(filename: str):
    # Prevent path traversal
    safe_name = Path(filename).name
    path = Path(OUTPUT_DIR) / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    mime = "audio/mpeg" if safe_name.endswith(".mp3") else "audio/wav"
    return FileResponse(str(path), media_type=mime)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=False)
