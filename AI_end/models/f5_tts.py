import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

from .worker_base import WorkerModel

logger = logging.getLogger(__name__)

_WORKER_SCRIPT = str(Path(__file__).parent.parent / "workers" / "f5_worker.py")
_PYTHON_EXE    = str(Path(__file__).parent.parent.parent / "ttsenv" / "Scripts" / "python.exe")
_OUTPUT_DIR    = str(Path(__file__).parent.parent / "outputs")

# Stub fallback (edge-tts) when worker is not running
def _edge_stub(text: str, out_path: str) -> None:
    import edge_tts
    async def _run():
        communicate = edge_tts.Communicate(text, "zh-TW-HsiaoChenNeural")
        await communicate.save(out_path)
    asyncio.run(_run())


class F5TTSModel(WorkerModel):
    def __init__(self):
        super().__init__(
            name="F5-TTS",
            python_exe=_PYTHON_EXE,
            worker_script=_WORKER_SCRIPT,
        )
        os.makedirs(_OUTPUT_DIR, exist_ok=True)

    def generate(self, text: str, ref_audio_path: str | None = None) -> dict:
        if self.is_loaded():
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                self.generate_async(text, ref_audio_path)
            )
            return {
                "audio_path":     result["audio_path"],
                "inference_time": result["inference_time"],
                "rtf":            result["rtf"],
            }
        # Stub fallback
        logger.warning("[F5-TTS] Worker not running — returning stub output")
        out_path = os.path.join(_OUTPUT_DIR, f"f5_{uuid.uuid4().hex}.mp3")
        start = time.perf_counter()
        _edge_stub(text, out_path)
        elapsed = round(time.perf_counter() - start, 3)
        return {"audio_path": out_path, "inference_time": elapsed,
                "rtf": round(elapsed / max(1.0, len(text) * 0.15), 3)}
