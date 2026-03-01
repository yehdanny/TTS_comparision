"""
Base class for persistent subprocess TTS workers.
Each model runs in its own Python venv via a long-lived subprocess.
Communication: newline-delimited JSON on stdin/stdout.
"""
import asyncio
import json
import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

WORKER_TIMEOUT = 300  # seconds to wait for a single inference


class WorkerModel:
    """
    Manages a single persistent worker subprocess.
    Call `await start()` once at startup; then `await generate(...)` per request.
    """

    def __init__(self, name: str, python_exe: str, worker_script: str):
        self._name   = name
        self._python = python_exe
        self._script = worker_script
        self._proc: asyncio.subprocess.Process | None = None
        self._loaded = False
        self._lock   = asyncio.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch the worker process and wait for it to print READY."""
        logger.info("[%s] Starting worker: %s %s", self._name, self._python, self._script)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._python, self._script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"},
                limit=2**20,  # 1 MB – prevents LimitOverrunError on long model-load logs
            )
            # Stream stderr to our logger in the background
            asyncio.create_task(self._pipe_stderr())
            # Wait for READY signal (model loaded)
            await self._wait_ready()
            self._loaded = True
            logger.info("[%s] Worker ready (pid=%s)", self._name, self._proc.pid)
        except Exception as exc:
            logger.error("[%s] Worker failed to start: %s", self._name, exc)
            self._loaded = False

    async def _wait_ready(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            line = await asyncio.wait_for(
                self._proc.stdout.readline(), timeout=300
            )
            if not line:
                raise RuntimeError("Worker exited before sending READY")
            decoded = line.decode().strip()
            logger.info("[%s] worker: %s", self._name, decoded)
            if decoded in ("READY", "LOADING"):
                if decoded == "READY":
                    return

    async def _pipe_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        async for line in self._proc.stderr:
            logger.debug("[%s] stderr: %s", self._name, line.decode().rstrip())

    def is_loaded(self) -> bool:
        return self._loaded and self._proc is not None and self._proc.returncode is None

    # ── inference ─────────────────────────────────────────────────────────────

    async def generate_async(self, text: str, ref_audio_path: str | None = None, ref_text: str = "") -> dict:
        if not self.is_loaded():
            raise RuntimeError(f"{self._name} worker is not running")

        req_id = str(uuid.uuid4())
        payload = json.dumps({"id": req_id, "text": text, "ref_audio": ref_audio_path, "ref_text": ref_text})

        async with self._lock:   # one request at a time per worker
            assert self._proc and self._proc.stdin and self._proc.stdout
            self._proc.stdin.write((payload + "\n").encode())
            await self._proc.stdin.drain()

            while True:
                raw = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=WORKER_TIMEOUT
                )
                if not raw:
                    raise RuntimeError(f"{self._name} worker closed stdout unexpectedly")
                line = raw.decode().strip()
                if not line:
                    continue
                try:
                    resp = json.loads(line)
                    break
                except json.JSONDecodeError:
                    logger.debug("[%s] non-JSON stdout (skipped): %s", self._name, line)
                    continue
        if "error" in resp and resp["error"]:
            raise RuntimeError(resp["error"])
        return resp

    def generate(self, text: str, ref_audio_path: str | None = None) -> dict:
        """Sync wrapper — runs the async method in the running event loop."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.generate_async(text, ref_audio_path))
