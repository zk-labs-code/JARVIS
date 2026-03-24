"""Configuration manager for JARVIS."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("jarvis.config")

# Default configuration values
DEFAULTS: dict[str, Any] = {
    "general": {
        "name": "JARVIS",
        "version": "1.0.0",
        "log_level": "INFO",
        "data_dir": "data",
        "log_dir": "logs",
    },
    "voice": {
        "listener": {
            "engine": "vosk",
            "model_path": "data/models/vosk-model-small-en-us-0.15",
            "sample_rate": 16000,
            "energy_threshold": 300,
            "pause_threshold": 0.8,
            "wake_word": "jarvis",
        },
        "speaker": {
            "engine": "pyttsx3",
            "rate": 175,
            "volume": 0.9,
            "voice_id": None,
        },
    },
    "gesture": {
        "enabled": True,
        "camera_index": 0,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.5,
    },
    "ai": {
        "model_path": "data/models/llm",
        "model_file": None,
        "context_length": 4096,
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "gpu_layers": -1,
        "threads": 4,
    },
    "memory": {
        "db_path": "data/memory/jarvis_memory.db",
        "max_conversation_history": 50,
        "learning_enabled": True,
    },
    "ui": {
        "opacity": 0.85,
        "width": 420,
        "height": 600,
        "position": "top-right",
        "theme": "dark",
        "font_size": 13,
        "always_on_top": True,
    },
}


class Config:
    """Configuration manager that loads from YAML and provides dot-notation access."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize config from YAML file.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self._data: dict[str, Any] = {}
        self._config_path = Path(config_path)
        self._load(config_path)

    def _load(self, config_path: str) -> None:
        """Load configuration from YAML file, falling back to defaults.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self._data = self._deep_copy(DEFAULTS)

        path = Path(config_path)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
                self._deep_merge(self._data, user_config)
                logger.info(f"Configuration loaded from {path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}. Using defaults.")
        else:
            logger.info(f"No config file at {path}. Using defaults.")

        # Resolve relative paths
        base_dir = path.parent if path.exists() else Path.cwd()
        self._resolve_paths(base_dir)

    def _deep_copy(self, data: dict[str, Any]) -> dict[str, Any]:
        """Deep copy a dictionary.

        Args:
            data: Dictionary to copy.

        Returns:
            Deep copy of the dictionary.
        """
        result: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = self._deep_copy(value)
            elif isinstance(value, list):
                result[key] = list(value)
            else:
                result[key] = value
        return result

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> None:
        """Recursively merge override dict into base dict.

        Args:
            base: Base dictionary to merge into.
            override: Override dictionary with values to apply.
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _resolve_paths(self, base_dir: Path) -> None:
        """Resolve relative paths in config to absolute paths.

        Args:
            base_dir: Base directory for resolving relative paths.
        """
        path_keys = [
            ("general", "data_dir"),
            ("general", "log_dir"),
            ("voice", "listener", "model_path"),
            ("ai", "model_path"),
            ("memory", "db_path"),
        ]
        for keys in path_keys:
            value = self.get(*keys)
            if value and not os.path.isabs(value):
                resolved = str(base_dir / value)
                self.set(resolved, *keys)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a config value by nested keys.

        Args:
            *keys: Sequence of keys for nested lookup.
            default: Default value if key not found.

        Returns:
            The config value or default.
        """
        current = self._data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def set(self, value: Any, *keys: str) -> None:
        """Set a config value by nested keys.

        Args:
            value: Value to set.
            *keys: Sequence of keys for nested lookup.
        """
        current = self._data
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def save(self, path: str | None = None) -> None:
        """Save current configuration to YAML file.

        Args:
            path: Output path. Uses original config path if None.
        """
        save_path = Path(path) if path else self._config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {save_path}")

    @property
    def data(self) -> dict[str, Any]:
        """Get the raw configuration dictionary."""
        return self._data
