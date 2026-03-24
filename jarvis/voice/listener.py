"""Voice recognition module for JARVIS.

Supports offline speech recognition using Vosk and online fallback
via Google Speech Recognition.
"""

import json
import logging
import queue
import threading
from typing import Any

logger = logging.getLogger("jarvis.voice.listener")


class VoiceListener:
    """Handles voice input using microphone with speech recognition."""

    def __init__(
        self,
        engine: str = "vosk",
        model_path: str = "",
        sample_rate: int = 16000,
        wake_word: str = "jarvis",
    ):
        """Initialize the voice listener.

        Args:
            engine: Recognition engine ('vosk' or 'google').
            model_path: Path to Vosk model directory.
            sample_rate: Audio sample rate.
            wake_word: Wake word to activate listening.
        """
        self.engine = engine
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.wake_word = wake_word.lower()
        self._running = False
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._recognizer: Any = None
        self._microphone: Any = None
        self._vosk_model: Any = None
        self._vosk_recognizer: Any = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the speech recognition system.

        Returns:
            True if initialization succeeded.
        """
        if self.engine == "vosk":
            return self._init_vosk()
        return self._init_google()

    def _init_vosk(self) -> bool:
        """Initialize Vosk offline speech recognition.

        Returns:
            True if successful.
        """
        try:
            from pathlib import Path

            import vosk
            model_path = Path(self.model_path)

            if not model_path.exists():
                logger.warning(
                    f"Vosk model not found at {model_path}. "
                    "Download from https://alphacephei.com/vosk/models "
                    "and extract to the model path. Falling back to Google."
                )
                return self._init_google()

            vosk.SetLogLevel(-1)  # Suppress Vosk logging
            self._vosk_model = vosk.Model(str(model_path))
            self._vosk_recognizer = vosk.KaldiRecognizer(
                self._vosk_model, self.sample_rate
            )
            self._initialized = True
            logger.info("Vosk speech recognition initialized")
            return True

        except ImportError:
            logger.warning("Vosk not installed. Falling back to Google recognition.")
            return self._init_google()
        except Exception as e:
            logger.error(f"Vosk initialization failed: {e}")
            return self._init_google()

    def _init_google(self) -> bool:
        """Initialize Google Speech Recognition (online fallback).

        Returns:
            True if successful.
        """
        try:
            import speech_recognition as sr

            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.pause_threshold = 0.8
            self._recognizer.dynamic_energy_threshold = True
            self.engine = "google"
            self._initialized = True
            logger.info("Google Speech Recognition initialized (online)")
            return True

        except ImportError:
            logger.error(
                "SpeechRecognition not installed. "
                "Install with: pip install SpeechRecognition"
            )
            return False

    def listen(self, timeout: int | None = None) -> str | None:
        """Listen for speech and return recognized text.

        Args:
            timeout: Maximum time to listen in seconds. None for default.

        Returns:
            Recognized text string or None if nothing recognized.
        """
        if not self._initialized:
            if not self.initialize():
                return None

        if self.engine == "vosk":
            return self._listen_vosk(timeout)
        return self._listen_google(timeout)

    def _listen_vosk(self, timeout: int | None = None) -> str | None:
        """Listen using Vosk offline recognition.

        Args:
            timeout: Maximum listen time in seconds.

        Returns:
            Recognized text or None.
        """
        try:
            import pyaudio

            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=4096,
            )

            logger.debug("Listening (Vosk)...")
            frames_read = 0
            max_frames = (self.sample_rate * (timeout or 10)) // 4096

            while frames_read < max_frames:
                data = stream.read(4096, exception_on_overflow=False)
                if self._vosk_recognizer.AcceptWaveform(data):
                    result = json.loads(self._vosk_recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        stream.stop_stream()
                        stream.close()
                        audio.terminate()
                        logger.info(f"Recognized (Vosk): {text}")
                        return text
                frames_read += 1

            # Get partial result
            result = json.loads(self._vosk_recognizer.FinalResult())
            text = result.get("text", "").strip()

            stream.stop_stream()
            stream.close()
            audio.terminate()

            if text:
                logger.info(f"Recognized (Vosk partial): {text}")
                return text

        except ImportError:
            logger.error("PyAudio not installed. Install with: pip install pyaudio")
        except Exception as e:
            logger.error(f"Vosk listening error: {e}")

        return None

    def _listen_google(self, timeout: int | None = None) -> str | None:
        """Listen using Google Speech Recognition.

        Args:
            timeout: Maximum listen time in seconds.

        Returns:
            Recognized text or None.
        """
        try:
            import speech_recognition as sr

            with sr.Microphone() as source:
                logger.debug("Listening (Google)...")
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self._recognizer.listen(
                    source,
                    timeout=timeout or 10,
                    phrase_time_limit=15,
                )

            try:
                text = self._recognizer.recognize_google(audio)
                logger.info(f"Recognized (Google): {text}")
                return text
            except sr.UnknownValueError:
                logger.debug("Could not understand audio")
            except sr.RequestError as e:
                logger.warning(f"Google Speech API error: {e}")

        except Exception as e:
            logger.error(f"Google listening error: {e}")

        return None

    def start_continuous(self, callback: Any) -> None:
        """Start continuous listening in a background thread.

        Args:
            callback: Function to call with recognized text.
        """
        self._running = True
        self._listen_thread = threading.Thread(
            target=self._continuous_loop,
            args=(callback,),
            daemon=True,
        )
        self._listen_thread.start()
        logger.info("Continuous listening started")

    def _continuous_loop(self, callback: Any) -> None:
        """Background loop for continuous listening.

        Args:
            callback: Function to call with recognized text.
        """
        while self._running:
            text = self.listen(timeout=5)
            if text:
                callback(text)

    def stop(self) -> None:
        """Stop listening."""
        self._running = False
        logger.info("Voice listener stopped")

    @property
    def is_initialized(self) -> bool:
        """Check if the listener is initialized."""
        return self._initialized
