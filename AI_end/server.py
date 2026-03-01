"""
TTS Comparison — FastAPI backend
Run: python server.py  (from AI_end/)
"""

import base64
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import MODEL_PATHS, OUTPUT_DIR, PORT

DEFAULT_REF_AUDIO = str(Path(__file__).parent.parent / "ref_audio" / "reference.wav")

SAMPLE_TEXTS = {
    "zh": "人工智慧語音合成技術正在快速發展，未來將有更多創新的應用場景。",
    "en": "Artificial intelligence is transforming the way we interact with technology every day.",
}
from models.f5_tts import F5TTSModel
from models.cosyvoice import CosyVoiceModel
from models.gpt_sovits import GPTSoVITsModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model instances ───────────────────────────────────────────────────────────
models = {
    "f5":        F5TTSModel(),
    "cosyvoice": CosyVoiceModel(),
    "gptsovits": GPTSoVITsModel(),
}


# ── Lifespan: start all workers on server startup ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    logger.info("Starting TTS workers (this may take 1-3 minutes for model loading)...")
    tasks = [m.start() for m in models.values()]
    await asyncio.gather(*tasks, return_exceptions=True)
    loaded = [k for k, m in models.items() if m.is_loaded()]
    stub   = [k for k, m in models.items() if not m.is_loaded()]
    if loaded:
        logger.info("Workers loaded: %s", loaded)
    if stub:
        logger.warning("Workers using stub (edge-tts fallback): %s", stub)
    yield
    # Cleanup: terminate worker subprocesses
    for m in models.values():
        if hasattr(m, "_proc") and m._proc:
            try:
                m._proc.terminate()
            except Exception:
                pass


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="TTS Comparison API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    reference_audio: str | None = None   # base64-encoded WAV
    ref_text: str = ""                   # transcript of reference audio (required by CosyVoice)


class TTSResponse(BaseModel):
    model: str
    audio_url: str
    inference_time: float
    rtf: float
    file_size_kb: float


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save_ref_audio(b64_data: str) -> str:
    raw = base64.b64decode(b64_data)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", dir=OUTPUT_DIR, delete=False)
    tmp.write(raw)
    tmp.close()
    return tmp.name


def _build_response(model_name: str, result: dict) -> TTSResponse:
    audio_path = result["audio_path"]
    size_kb    = round(os.path.getsize(audio_path) / 1024, 1)
    filename   = Path(audio_path).name
    return TTSResponse(
        model=model_name,
        audio_url=f"/api/audio/{filename}",
        inference_time=round(result["inference_time"], 3),
        rtf=round(result["rtf"], 3),
        file_size_kb=size_kb,
    )


async def _run_model(key: str, req: TTSRequest) -> TTSResponse:
    model    = models[key]
    ref_path = None
    if req.reference_audio:
        try:
            ref_path = _save_ref_audio(req.reference_audio)
        except Exception as exc:
            logger.warning("Could not decode reference audio: %s", exc)
    try:
        if model.is_loaded():
            result = await model.generate_async(req.text, ref_path, req.ref_text)
        else:
            import asyncio
            result = await asyncio.get_event_loop().run_in_executor(
                None, model.generate, req.text, ref_path
            )
    except Exception as exc:
        logger.error("[%s] generation error: %s", key, exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
    return _build_response(key, result)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models": {
            k: {"loaded": m.is_loaded()} for k, m in models.items()
        },
    }


@app.post("/api/tts/f5", response_model=TTSResponse)
async def tts_f5(req: TTSRequest):
    return await _run_model("f5", req)


@app.post("/api/tts/cosyvoice", response_model=TTSResponse)
async def tts_cosyvoice(req: TTSRequest):
    return await _run_model("cosyvoice", req)


@app.post("/api/tts/gptsovits", response_model=TTSResponse)
async def tts_gptsovits(req: TTSRequest):
    return await _run_model("gptsovits", req)


@app.post("/api/regenerate-samples")
async def regenerate_samples():
    if not Path(DEFAULT_REF_AUDIO).exists():
        raise HTTPException(status_code=400, detail="No reference audio found at ref_audio/reference.wav")
    urls = {}
    for lang, text in SAMPLE_TEXTS.items():
        for model_key, model in models.items():
            if not model.is_loaded():
                continue
            try:
                result = await model.generate_async(text, DEFAULT_REF_AUDIO)
                fixed_name = f"sample_{lang}_{model_key}.wav"
                fixed_path = str(Path(OUTPUT_DIR) / fixed_name)
                shutil.copy2(result["audio_path"], fixed_path)
                urls[f"{lang}_{model_key}"] = f"/api/audio/{fixed_name}"
            except Exception as exc:
                logger.error("[regenerate-samples] %s/%s failed: %s", lang, model_key, exc)
    return {"status": "ok", "urls": urls}


@app.get("/api/audio/{filename}")
def serve_audio(filename: str):
    safe_name = Path(filename).name
    path      = Path(OUTPUT_DIR) / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    mime = "audio/mpeg" if safe_name.endswith(".mp3") else "audio/wav"
    return FileResponse(str(path), media_type=mime)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=False)
