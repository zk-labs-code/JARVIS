"""Text-to-Speech module for JARVIS.

Provides offline speech synthesis using pyttsx3.
"""

import logging
import threading
from typing import Any

logger = logging.getLogger("jarvis.voice.speaker")


class Speaker:
    """Text-to-speech engine for JARVIS voice output."""

    def __init__(
        self,
        rate: int = 175,
        volume: float = 0.9,
        voice_id: str | None = None,
    ):
        """Initialize the speaker.

        Args:
            rate: Speech rate (words per minute).
            volume: Volume level (0.0 to 1.0).
            voice_id: Specific voice ID to use. None for system default.
        """
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id
        self._engine: Any = None
        self._lock = threading.Lock()
        self._initialized = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the TTS engine."""
        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)

            if self.voice_id:
                self._engine.setProperty("voice", self.voice_id)

            self._initialized = True
            logger.info(f"TTS engine initialized (rate={self.rate}, volume={self.volume})")

        except ImportError:
            logger.warning(
                "pyttsx3 not installed. TTS disabled. "
                "Install with: pip install pyttsx3"
            )
        except Exception as e:
            logger.warning(f"TTS initialization failed: {e}. Speech output disabled.")

    def speak(self, text: str, block: bool = False) -> None:
        """Speak the given text.

        Args:
            text: Text to speak.
            block: If True, block until speech completes.
        """
        if not self._initialized or not text.strip():
            logger.info(f"[TTS disabled] Would say: {text}")
            return

        if block:
            self._speak_sync(text)
        else:
            thread = threading.Thread(
                target=self._speak_sync, args=(text,), daemon=True
            )
            thread.start()

    def _speak_sync(self, text: str) -> None:
        """Speak text synchronously (thread-safe).

        Args:
            text: Text to speak.
        """
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")

    def get_available_voices(self) -> list[dict[str, str]]:
        """Get list of available TTS voices.

        Returns:
            List of voice info dictionaries.
        """
        if not self._initialized:
            return []

        try:
            voices = self._engine.getProperty("voices")
            return [
                {
                    "id": voice.id,
                    "name": voice.name,
                    "languages": str(voice.languages),
                }
                for voice in voices
            ]
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return []

    def set_voice(self, voice_id: str) -> None:
        """Change the TTS voice.

        Args:
            voice_id: Voice identifier string.
        """
        if not self._initialized:
            return
        try:
            self._engine.setProperty("voice", voice_id)
            self.voice_id = voice_id
            logger.info(f"Voice changed to: {voice_id}")
        except Exception as e:
            logger.error(f"Failed to set voice: {e}")

    def set_rate(self, rate: int) -> None:
        """Change speech rate.

        Args:
            rate: Words per minute.
        """
        if not self._initialized:
            return
        self._engine.setProperty("rate", rate)
        self.rate = rate

    def set_volume(self, volume: float) -> None:
        """Change volume level.

        Args:
            volume: Volume (0.0 to 1.0).
        """
        if not self._initialized:
            return
        self._engine.setProperty("volume", max(0.0, min(1.0, volume)))
        self.volume = volume

    @property
    def is_initialized(self) -> bool:
        """Check if TTS is initialized."""
        return self._initialized
