"""Main JARVIS Engine - Orchestrates all subsystems.

The engine is the central coordinator that connects voice recognition,
gesture detection, AI brain, action execution, and the UI overlay.
"""

import logging
import threading
import time
from typing import Any, Callable

from jarvis.actions.coding import CodingAssistant
from jarvis.actions.system import SystemController
from jarvis.actions.web import WebAssistant
from jarvis.core.brain import Brain
from jarvis.core.config import Config
from jarvis.core.memory import Memory
from jarvis.skills.skill_manager import SkillManager
from jarvis.utils.gpu import GPUManager
from jarvis.vision.gesture import GestureRecognizer
from jarvis.voice.listener import VoiceListener
from jarvis.voice.speaker import Speaker

logger = logging.getLogger("jarvis.engine")


class JarvisEngine:
    """Main engine that orchestrates all JARVIS subsystems."""

    def __init__(self, config: Config):
        """Initialize the JARVIS engine.

        Args:
            config: Configuration instance.
        """
        self.config = config
        self._running = False
        self._listening = False
        self._lock = threading.Lock()

        # Callbacks for UI updates
        self._on_status_change: Callable[[str], None] | None = None
        self._on_response: Callable[[str, str], None] | None = None
        self._on_listening_change: Callable[[bool], None] | None = None

        # Initialize subsystems
        self._init_subsystems()

    def _init_subsystems(self) -> None:
        """Initialize all JARVIS subsystems."""
        logger.info("Initializing JARVIS subsystems...")

        # GPU Manager
        self.gpu = GPUManager(
            prefer_gpu=self.config.get("gpu", "prefer_gpu", default=True),
            memory_fraction=self.config.get("gpu", "memory_fraction", default=0.8),
        )
        self.gpu.setup_environment()

        # Memory System
        self.memory = Memory(
            db_path=self.config.get("memory", "db_path"),
            max_history=self.config.get("memory", "max_conversation_history", default=50),
        )

        # Determine GPU backend (auto-detect or from config)
        gpu_backend = self.config.get("gpu", "backend", default="auto")
        if gpu_backend == "auto":
            gpu_backend = self.gpu.get_llama_cpp_backend()

        # AI Brain
        self.brain = Brain(
            model_path=self.config.get("ai", "model_path"),
            model_file=self.config.get("ai", "model_file"),
            context_length=self.config.get("ai", "context_length", default=4096),
            max_tokens=self.config.get("ai", "max_tokens", default=2048),
            temperature=self.config.get("ai", "temperature", default=0.7),
            top_p=self.config.get("ai", "top_p", default=0.9),
            gpu_layers=self.gpu.get_gpu_layers() if self.gpu.is_available
            else self.config.get("ai", "gpu_layers", default=0),
            threads=self.config.get("ai", "threads", default=4),
            gpu_backend=gpu_backend,
        )

        # Voice
        self.listener = VoiceListener(
            engine=self.config.get("voice", "listener", "engine", default="vosk"),
            model_path=self.config.get("voice", "listener", "model_path"),
            sample_rate=self.config.get("voice", "listener", "sample_rate", default=16000),
            wake_word=self.config.get("voice", "listener", "wake_word", default="jarvis"),
        )
        self.speaker = Speaker(
            rate=self.config.get("voice", "speaker", "rate", default=175),
            volume=self.config.get("voice", "speaker", "volume", default=0.9),
        )

        # Gesture Recognition
        self.gesture = GestureRecognizer(
            camera_index=self.config.get("gesture", "camera_index", default=0),
            min_detection_confidence=self.config.get(
                "gesture", "min_detection_confidence", default=0.7
            ),
            min_tracking_confidence=self.config.get(
                "gesture", "min_tracking_confidence", default=0.5
            ),
        )

        # Action Handlers
        self.system_controller = SystemController()
        self.web_assistant = WebAssistant(
            timeout=self.config.get("web", "timeout", default=15),
            max_results=self.config.get("web", "max_search_results", default=5),
        )
        self.coding_assistant = CodingAssistant(
            max_execution_time=self.config.get("coding", "max_execution_time", default=30),
        )

        # Skill Manager
        self.skill_manager = SkillManager(
            memory=self.memory,
            web_assistant=self.web_assistant,
        )

        logger.info("All subsystems initialized")

    def start(self) -> None:
        """Start the JARVIS engine and all subsystems."""
        if self._running:
            logger.warning("Engine already running")
            return

        logger.info("Starting JARVIS Engine...")
        self._notify_status("Initializing AI brain...")

        # Initialize Brain
        self.brain.initialize()
        if self.brain.is_fallback:
            self._notify_status("Running in fallback mode (no LLM model)")
        else:
            self._notify_status("AI brain ready")

        self._running = True

        # Start voice listener in background thread
        self._voice_thread = threading.Thread(
            target=self._voice_loop, daemon=True, name="voice-listener"
        )
        self._voice_thread.start()

        # Start gesture recognition if enabled
        if self.config.get("gesture", "enabled", default=True):
            self._gesture_thread = threading.Thread(
                target=self._gesture_loop, daemon=True, name="gesture-recognizer"
            )
            self._gesture_thread.start()

        self._notify_status("JARVIS is ready")
        self.speaker.speak("JARVIS online. How can I help you?")
        logger.info("JARVIS Engine started successfully")

    def stop(self) -> None:
        """Stop the JARVIS engine and all subsystems."""
        logger.info("Stopping JARVIS Engine...")
        self._running = False
        self._listening = False

        self.listener.stop()
        self.gesture.stop()
        self.speaker.speak("JARVIS shutting down. Goodbye.")

        self._notify_status("JARVIS offline")
        logger.info("JARVIS Engine stopped")

    def process_text_input(self, text: str) -> str:
        """Process a text command (from UI input or voice).

        Args:
            text: User input text.

        Returns:
            Response string.
        """
        if not text.strip():
            return ""

        logger.info(f"Processing input: {text}")
        self._notify_status("Thinking...")

        # Store in memory
        self.memory.add_conversation("user", text)

        # Get AI response
        history = self.memory.get_conversation_history()
        response = self.brain.think(text, conversation_history=history)

        # Parse and execute action
        action_type, action_params = self.brain.parse_action(response)
        result = self._execute_action(action_type, action_params, text)

        # Store response
        self.memory.add_conversation("assistant", result)
        self.memory.log_command(text, action_type, success=True)

        # Notify UI
        self._notify_response(text, result)
        self._notify_status("Ready")

        return result

    def _execute_action(self, action_type: str, params: str, original_input: str) -> str:
        """Execute a parsed action.

        Args:
            action_type: Type of action to execute.
            params: Action parameters.
            original_input: Original user input.

        Returns:
            Result string.
        """
        logger.info(f"Executing action: {action_type} -> {params}")

        try:
            if action_type == "SYSTEM_CMD":
                return self.system_controller.execute(params)
            elif action_type == "CODE_RUN":
                return self.coding_assistant.execute(params)
            elif action_type == "WEB_SEARCH":
                return self.web_assistant.search(params)
            elif action_type == "FILE_OP":
                return self.system_controller.file_operation(params)
            elif action_type == "APP_CONTROL":
                return self.system_controller.app_control(params)
            elif action_type == "LEARN":
                return self.skill_manager.learn_skill(params)
            elif action_type == "ANSWER":
                return params
            else:
                return params
        except Exception as e:
            error_msg = f"Error executing {action_type}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _voice_loop(self) -> None:
        """Background loop for continuous voice listening."""
        logger.info("Voice listener started")
        wake_word = self.config.get("voice", "listener", "wake_word", default="jarvis")

        while self._running:
            try:
                # Listen for wake word or continuous listening
                text = self.listener.listen()
                if text is None:
                    continue

                text_lower = text.lower().strip()

                # Check for wake word
                if wake_word in text_lower:
                    # Remove wake word from command
                    command = text_lower.replace(wake_word, "").strip()
                    if command:
                        self._notify_listening(True)
                        self.speaker.speak("Yes?")
                        response = self.process_text_input(command)
                        self.speaker.speak(self._clean_for_speech(response))
                        self._notify_listening(False)
                    else:
                        self._notify_listening(True)
                        self.speaker.speak("I'm listening.")
                        # Wait for follow-up command
                        follow_up = self.listener.listen(timeout=10)
                        if follow_up:
                            response = self.process_text_input(follow_up)
                            self.speaker.speak(self._clean_for_speech(response))
                        self._notify_listening(False)

                elif self._listening:
                    # Continuous listening mode
                    response = self.process_text_input(text)
                    self.speaker.speak(self._clean_for_speech(response))

            except Exception as e:
                logger.error(f"Voice loop error: {e}")
                time.sleep(1)

    def _gesture_loop(self) -> None:
        """Background loop for gesture recognition."""
        logger.info("Gesture recognizer started")
        gesture_actions = self.config.get("gesture", "gestures", default={})

        while self._running:
            try:
                gesture = self.gesture.detect()
                if gesture is None:
                    continue

                logger.info(f"Gesture detected: {gesture}")
                action = gesture_actions.get(gesture)

                if action:
                    self._handle_gesture_action(action, gesture)

            except Exception as e:
                logger.error(f"Gesture loop error: {e}")
                time.sleep(1)

    def _handle_gesture_action(self, action: str, gesture: str) -> None:
        """Handle a gesture-triggered action.

        Args:
            action: The action to perform.
            gesture: The gesture that triggered it.
        """
        if action == "activate_listening":
            self._listening = True
            self._notify_listening(True)
            self.speaker.speak("Listening mode activated.")
        elif action == "stop_listening":
            self._listening = False
            self._notify_listening(False)
            self.speaker.speak("Listening mode deactivated.")
        elif action == "confirm_action":
            self.speaker.speak("Confirmed.")
        elif action == "cancel_action":
            self.speaker.speak("Cancelled.")
        elif action == "screenshot":
            result = self.system_controller.execute("screenshot")
            self.speaker.speak(result)
        elif action in ("scroll_up", "scroll_down"):
            self.system_controller.execute(action)
        else:
            # Treat as a text command
            self.process_text_input(action)

    def _clean_for_speech(self, text: str) -> str:
        """Clean response text for text-to-speech output.

        Args:
            text: Raw response text.

        Returns:
            Cleaned text suitable for speech.
        """
        # Remove action tags
        if "[ACTION:" in text:
            _, text = self.brain.parse_action(text)

        # Truncate long responses for speech
        if len(text) > 500:
            text = text[:500] + "... I've displayed the full response on screen."

        return text

    # --- UI Callback Management ---

    def set_status_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for status updates.

        Args:
            callback: Function to call with status string.
        """
        self._on_status_change = callback

    def set_response_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set callback for response updates.

        Args:
            callback: Function to call with (input, response) strings.
        """
        self._on_response = callback

    def set_listening_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback for listening state changes.

        Args:
            callback: Function to call with listening state.
        """
        self._on_listening_change = callback

    def _notify_status(self, status: str) -> None:
        """Notify status change.

        Args:
            status: Status message.
        """
        logger.info(f"Status: {status}")
        if self._on_status_change:
            self._on_status_change(status)

    def _notify_response(self, user_input: str, response: str) -> None:
        """Notify of new response.

        Args:
            user_input: User's input.
            response: JARVIS's response.
        """
        if self._on_response:
            self._on_response(user_input, response)

    def _notify_listening(self, listening: bool) -> None:
        """Notify of listening state change.

        Args:
            listening: Whether JARVIS is actively listening.
        """
        self._listening = listening
        if self._on_listening_change:
            self._on_listening_change(listening)

    @property
    def is_running(self) -> bool:
        """Check if the engine is running."""
        return self._running

    def get_system_info(self) -> dict[str, Any]:
        """Get comprehensive system information.

        Returns:
            Dictionary with system status information.
        """
        return {
            "engine_running": self._running,
            "listening": self._listening,
            "brain": self.brain.get_status(),
            "gpu": self.gpu.get_status_summary(),
            "memory": self.memory.get_stats(),
        }
