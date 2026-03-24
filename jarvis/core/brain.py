"""AI Brain module for JARVIS.

Handles local LLM inference using llama-cpp-python with GPU acceleration.
Provides intelligent response generation, command interpretation, and
natural language understanding.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.brain")

# System prompt that defines JARVIS's personality and capabilities
SYSTEM_PROMPT = """You are JARVIS, an advanced AI personal assistant. You are running locally
on the user's computer with full system access. You can:

1. Execute system commands and control the computer
2. Write, run, and debug code in multiple languages
3. Search the internet and learn new skills
4. Manage files and applications
5. Answer questions and provide information
6. Control windows, applications, and system settings

Guidelines:
- Be concise and direct in responses
- When asked to perform an action, respond with the action type and parameters
- Format action responses as: [ACTION:type] parameters
- Available action types: SYSTEM_CMD, CODE_RUN, WEB_SEARCH, FILE_OP, APP_CONTROL, ANSWER, LEARN
- If you need clarification, ask briefly
- For coding tasks, provide complete, runnable code
- Always prioritize safety - warn before destructive operations

Examples of action responses:
- User: "Open notepad" -> [ACTION:SYSTEM_CMD] open notepad
- User: "Search for Python tutorials" -> [ACTION:WEB_SEARCH] Python tutorials
- User: "Write a hello world in Python" -> [ACTION:CODE_RUN] print("Hello, World!")
- User: "What time is it?" -> [ACTION:SYSTEM_CMD] get_time
- User: "What is machine learning?" -> [ACTION:ANSWER] Machine learning is...
"""


class Brain:
    """AI Brain using local LLM for natural language understanding and response generation."""

    def __init__(
        self,
        model_path: str,
        model_file: str | None = None,
        context_length: int = 4096,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        gpu_layers: int = -1,
        threads: int = 4,
    ):
        """Initialize the AI Brain.

        Args:
            model_path: Path to the model directory.
            model_file: Specific model file name (auto-detected if None).
            context_length: Maximum context length for the model.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            gpu_layers: Number of layers to offload to GPU (-1 for all).
            threads: Number of CPU threads to use.
        """
        self.model_path = model_path
        self.model_file = model_file
        self.context_length = context_length
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.gpu_layers = gpu_layers
        self.threads = threads
        self._llm: Any = None
        self._initialized = False
        self._fallback_mode = False

    def initialize(self) -> bool:
        """Initialize the LLM model.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        model_file_path = self._find_model()
        if model_file_path is None:
            logger.warning(
                "No LLM model found. Brain will operate in fallback mode "
                "(rule-based responses). Download a GGUF model to: "
                f"{self.model_path}"
            )
            self._fallback_mode = True
            self._initialized = True
            return True

        try:
            from llama_cpp import Llama

            logger.info(f"Loading LLM model: {model_file_path}")
            logger.info(f"GPU layers: {self.gpu_layers}, Threads: {self.threads}")

            self._llm = Llama(
                model_path=str(model_file_path),
                n_ctx=self.context_length,
                n_gpu_layers=self.gpu_layers,
                n_threads=self.threads,
                verbose=False,
            )
            self._initialized = True
            self._fallback_mode = False
            logger.info("LLM model loaded successfully")
            return True

        except ImportError:
            logger.warning(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-cpp-python"
            )
            self._fallback_mode = True
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self._fallback_mode = True
            self._initialized = True
            return True

    def _find_model(self) -> Path | None:
        """Find a GGUF model file in the model directory.

        Returns:
            Path to the model file or None.
        """
        if self.model_file:
            path = Path(self.model_path) / self.model_file
            if path.exists():
                return path

        model_dir = Path(self.model_path)
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            return None

        # Search for GGUF files
        gguf_files = list(model_dir.glob("*.gguf"))
        if gguf_files:
            # Prefer smaller quantized models for faster loading
            for preferred in ["q4_k_m", "q4_0", "q5_k_m", "q8_0"]:
                for f in gguf_files:
                    if preferred in f.name.lower():
                        return f
            return gguf_files[0]

        return None

    def think(
        self,
        user_input: str,
        conversation_history: list[dict[str, str]] | None = None,
        context: str | None = None,
    ) -> str:
        """Process user input and generate a response.

        Args:
            user_input: The user's message or command.
            conversation_history: Previous conversation for context.
            context: Additional context information.

        Returns:
            Generated response string.
        """
        if not self._initialized:
            return "[ACTION:ANSWER] Brain not initialized. Please wait..."

        if self._fallback_mode:
            return self._fallback_think(user_input)

        return self._llm_think(user_input, conversation_history, context)

    def _llm_think(
        self,
        user_input: str,
        conversation_history: list[dict[str, str]] | None = None,
        context: str | None = None,
    ) -> str:
        """Generate response using the LLM.

        Args:
            user_input: User message.
            conversation_history: Previous conversation.
            context: Additional context.

        Returns:
            LLM-generated response.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context:
            messages.append({
                "role": "system",
                "content": f"Additional context: {context}",
            })

        # Add conversation history
        if conversation_history:
            recent = conversation_history[-10:]  # Last 10 exchanges
            messages.extend(recent)

        messages.append({"role": "user", "content": user_input})

        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=["User:", "\n\n\n"],
            )

            content = response["choices"][0]["message"]["content"]
            return content.strip() if content else "[ACTION:ANSWER] I couldn't generate a response."

        except Exception as e:
            logger.error(f"LLM inference error: {e}")
            return self._fallback_think(user_input)

    def _fallback_think(self, user_input: str) -> str:
        """Rule-based fallback when LLM is not available.

        Args:
            user_input: User message.

        Returns:
            Rule-based response.
        """
        text = user_input.lower().strip()

        # System commands
        if any(w in text for w in ["open ", "launch ", "start ", "run "]):
            app = text.split(maxsplit=1)[1] if " " in text else ""
            return f"[ACTION:SYSTEM_CMD] open {app}"

        if any(w in text for w in ["close ", "kill ", "stop ", "quit "]):
            app = text.split(maxsplit=1)[1] if " " in text else ""
            return f"[ACTION:SYSTEM_CMD] close {app}"

        # Web search
        if any(w in text for w in ["search ", "look up ", "find ", "google "]):
            query = text.split(maxsplit=1)[1] if " " in text else text
            return f"[ACTION:WEB_SEARCH] {query}"

        # Code execution
        if any(w in text for w in ["write code", "create script", "program", "code "]):
            return f"[ACTION:CODE_RUN] # Task: {user_input}"

        # File operations
        if any(w in text for w in ["create file", "delete file", "move file", "copy file",
                                    "rename file", "list files"]):
            return f"[ACTION:FILE_OP] {user_input}"

        # Screenshots
        if "screenshot" in text:
            return "[ACTION:SYSTEM_CMD] screenshot"

        # Time/date
        if any(w in text for w in ["time", "date", "day"]):
            return "[ACTION:SYSTEM_CMD] get_time"

        # Volume control
        if any(w in text for w in ["volume", "mute", "unmute"]):
            return f"[ACTION:SYSTEM_CMD] {text}"

        # Learning/self-improvement
        if any(w in text for w in ["learn ", "study ", "teach yourself"]):
            topic = text.split(maxsplit=1)[1] if " " in text else text
            return f"[ACTION:LEARN] {topic}"

        # Default: treat as a question
        return f"[ACTION:ANSWER] I understand you said: '{user_input}'. I'm in fallback mode (no LLM loaded). Please download a GGUF model for full AI capabilities."

    def parse_action(self, response: str) -> tuple[str, str]:
        """Parse an action response from the brain.

        Args:
            response: Response string from think().

        Returns:
            Tuple of (action_type, action_params).
        """
        if "[ACTION:" in response:
            try:
                action_start = response.index("[ACTION:") + 8
                action_end = response.index("]", action_start)
                action_type = response[action_start:action_end].strip()
                action_params = response[action_end + 1:].strip()
                return action_type, action_params
            except (ValueError, IndexError):
                pass

        return "ANSWER", response

    @property
    def is_initialized(self) -> bool:
        """Check if the brain is initialized."""
        return self._initialized

    @property
    def is_fallback(self) -> bool:
        """Check if brain is in fallback mode."""
        return self._fallback_mode

    def get_status(self) -> dict[str, Any]:
        """Get brain status information.

        Returns:
            Status dictionary.
        """
        return {
            "initialized": self._initialized,
            "fallback_mode": self._fallback_mode,
            "model_path": self.model_path,
            "gpu_layers": self.gpu_layers,
            "context_length": self.context_length,
        }
