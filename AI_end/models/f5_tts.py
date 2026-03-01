import asyncio
import logging
import os
import time
import uuid

import numpy as np
import soundfile as sf

from .base import BaseTTS

logger = logging.getLogger(__name__)

try:
    # Real import — uncomment and adjust once F5-TTS is installed:
    # from f5_tts.api import F5TTS as _F5TTS
    raise ImportError("F5-TTS not installed")
except ImportError:
    _F5TTS = None


class F5TTSModel(BaseTTS):
    def __init__(self, model_path: str = "", output_dir: str = "outputs"):
        self._loaded = False
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if _F5TTS is not None and model_path:
            try:
                self._model = _F5TTS(model_path)
                self._loaded = True
                logger.info("[F5-TTS] Model loaded from %s", model_path)
            except Exception as exc:
                logger.warning("[F5-TTS] Failed to load model: %s", exc)
        else:
            logger.warning("[F5-TTS] Model not loaded — returning stub output")

    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, text: str, ref_audio_path: str | None = None) -> dict:
        if self._loaded:
            return self._real_generate(text, ref_audio_path)
        return self._stub_generate(text)

    def _real_generate(self, text: str, ref_audio_path: str | None) -> dict:
        start = time.perf_counter()
        out_path = os.path.join(self._output_dir, f"f5_{uuid.uuid4().hex}.wav")
        # TODO: call self._model.infer(text, ref_audio=ref_audio_path, out_path=out_path)
        elapsed = time.perf_counter() - start
        duration = _wav_duration(out_path)
        return {
            "audio_path": out_path,
            "inference_time": elapsed,
            "rtf": elapsed / duration if duration else 0.0,
        }

    def _stub_generate(self, text: str) -> dict:
        logger.warning("[F5-TTS] Model not loaded — returning stub output")
        out_path = os.path.join(self._output_dir, f"f5_{uuid.uuid4().hex}.mp3")
        start = time.perf_counter()
        # zh-TW-HsiaoChenNeural — Female, Traditional Chinese
        _edge_tts_synthesize(text, voice="zh-TW-HsiaoChenNeural", out_path=out_path)
        inference_time = round(time.perf_counter() - start, 3)
        estimated_duration = max(1.0, len(text) * 0.15)
        return {
            "audio_path": out_path,
            "inference_time": inference_time,
            "rtf": round(inference_time / estimated_duration, 3),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _edge_tts_synthesize(text: str, voice: str, out_path: str) -> None:
    import edge_tts

    async def _run() -> None:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out_path)

    asyncio.run(_run())


def _wav_duration(path: str) -> float:
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        return 1.0
