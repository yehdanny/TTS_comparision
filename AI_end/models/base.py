from abc import ABC, abstractmethod


class BaseTTS(ABC):
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model is loaded and ready."""
        ...

    @abstractmethod
    def generate(self, text: str, ref_audio_path: str | None = None) -> dict:
        """
        Run TTS inference.

        Returns:
            {
                "audio_path": str,        # absolute path to generated WAV
                "inference_time": float,  # seconds
                "rtf": float,             # real-time factor
            }
        """
        ...
