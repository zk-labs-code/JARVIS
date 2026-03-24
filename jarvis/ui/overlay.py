"""Overlay UI module for JARVIS.

Provides a transparent, always-on-top overlay window using PyQt5
that displays JARVIS status, conversation, and controls.
"""

import logging
from typing import Any

logger = logging.getLogger("jarvis.ui.overlay")

# Check for PyQt5 availability
try:
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logger.warning("PyQt5 not installed. UI overlay disabled. Install with: pip install PyQt5")


class MessageBubble(QFrame):
    """A chat message bubble widget."""

    def __init__(self, text: str, is_user: bool = True, colors: dict[str, str] | None = None):
        """Initialize message bubble.

        Args:
            text: Message text.
            is_user: True if this is a user message.
            colors: Color theme dictionary.
        """
        super().__init__()
        colors = colors or {}
        bg = colors.get("primary", "#0f3460") if is_user else colors.get("surface", "#16213e")
        text_color = colors.get("text", "#eaeaea")

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg};
                border-radius: 12px;
                padding: 8px 12px;
                margin: 2px {'20px 2px 4px' if is_user else '4px 2px 20px'};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        role_label = QLabel("You" if is_user else "JARVIS")
        role_label.setStyleSheet(f"""
            color: {colors.get('accent', '#e94560') if not is_user else colors.get('text_secondary', '#a0a0a0')};
            font-size: 10px;
            font-weight: bold;
        """)
        layout.addWidget(role_label)

        msg_label = QLabel(text)
        msg_label.setWordWrap(True)
        msg_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        msg_label.setStyleSheet(f"""
            color: {text_color};
            font-size: 13px;
        """)
        layout.addWidget(msg_label)


class OverlayWindow(QMainWindow):
    """Transparent overlay window for JARVIS UI."""

    # Signals for thread-safe UI updates
    status_signal = pyqtSignal(str)
    response_signal = pyqtSignal(str, str)
    listening_signal = pyqtSignal(bool)

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the overlay window.

        Args:
            config: UI configuration dictionary.
        """
        super().__init__()
        self.config = config or {}
        self._engine: Any = None
        self._is_listening = False
        self._dragging = False
        self._drag_position = None

        # Get config values
        self.colors = self.config.get("colors", {
            "background": "#1a1a2e",
            "surface": "#16213e",
            "primary": "#0f3460",
            "accent": "#e94560",
            "text": "#eaeaea",
            "text_secondary": "#a0a0a0",
            "success": "#00c853",
            "warning": "#ff9100",
            "error": "#ff1744",
        })

        self._setup_window()
        self._create_ui()
        self._connect_signals()

    def _setup_window(self) -> None:
        """Configure window properties for overlay mode."""
        self.setWindowTitle("JARVIS")
        width = self.config.get("width", 420)
        height = self.config.get("height", 600)
        self.setFixedSize(width, height)

        # Window flags for overlay behavior
        flags = Qt.FramelessWindowHint | Qt.Tool
        if self.config.get("always_on_top", True):
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)

        # Transparency
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        opacity = self.config.get("opacity", 0.85)
        self.setWindowOpacity(opacity)

        # Position window
        self._position_window()

    def _position_window(self) -> None:
        """Position the window based on config."""
        position = self.config.get("position", "top-right")
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        screen_geo = screen.availableGeometry()
        w, h = self.width(), self.height()
        margin = 10

        positions = {
            "top-left": (margin, margin),
            "top-right": (screen_geo.width() - w - margin, margin),
            "bottom-left": (margin, screen_geo.height() - h - margin),
            "bottom-right": (screen_geo.width() - w - margin, screen_geo.height() - h - margin),
            "center": ((screen_geo.width() - w) // 2, (screen_geo.height() - h) // 2),
        }

        x, y = positions.get(position, positions["top-right"])
        self.move(x, y)

    def _create_ui(self) -> None:
        """Create the UI layout and widgets."""
        # Main container
        container = QWidget()
        container.setStyleSheet(f"""
            QWidget {{
                background-color: {self.colors['background']};
                border-radius: 16px;
            }}
        """)
        self.setCentralWidget(container)

        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # --- Title Bar ---
        title_bar = QHBoxLayout()

        # JARVIS logo/title
        title_label = QLabel("JARVIS")
        title_label.setStyleSheet(f"""
            color: {self.colors['accent']};
            font-size: 18px;
            font-weight: bold;
            font-family: 'Segoe UI', 'Arial', sans-serif;
        """)
        title_bar.addWidget(title_label)

        title_bar.addStretch()

        # Minimize button
        min_btn = QPushButton("—")
        min_btn.setFixedSize(28, 28)
        min_btn.setStyleSheet(self._button_style())
        min_btn.clicked.connect(self.showMinimized)
        title_bar.addWidget(min_btn)

        # Close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(28, 28)
        close_btn.setStyleSheet(self._button_style(self.colors["error"]))
        close_btn.clicked.connect(self._on_close)
        title_bar.addWidget(close_btn)

        main_layout.addLayout(title_bar)

        # --- Status Bar ---
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet(f"""
            color: {self.colors['warning']};
            font-size: 11px;
            padding: 4px 8px;
            background-color: {self.colors['surface']};
            border-radius: 8px;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        # --- Listening Indicator ---
        self.listening_indicator = QLabel("● Listening")
        self.listening_indicator.setStyleSheet(f"""
            color: {self.colors['success']};
            font-size: 12px;
            font-weight: bold;
        """)
        self.listening_indicator.setAlignment(Qt.AlignCenter)
        self.listening_indicator.setVisible(False)
        main_layout.addWidget(self.listening_indicator)

        # --- Chat Area ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                width: 6px;
                background: {self.colors['background']};
            }}
            QScrollBar::handle:vertical {{
                background: {self.colors['primary']};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)

        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(4)
        self.scroll_area.setWidget(self.chat_widget)

        main_layout.addWidget(self.scroll_area, stretch=1)

        # --- Input Area ---
        input_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a command or ask a question...")
        self.input_field.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.colors['surface']};
                color: {self.colors['text']};
                border: 1px solid {self.colors['primary']};
                border-radius: 12px;
                padding: 8px 14px;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border: 1px solid {self.colors['accent']};
            }}
        """)
        self.input_field.returnPressed.connect(self._on_send)
        input_layout.addWidget(self.input_field, stretch=1)

        # Send button
        send_btn = QPushButton("▶")
        send_btn.setFixedSize(38, 38)
        send_btn.setStyleSheet(self._button_style(self.colors["accent"]))
        send_btn.clicked.connect(self._on_send)
        input_layout.addWidget(send_btn)

        # Mic button
        self.mic_btn = QPushButton("🎤")
        self.mic_btn.setFixedSize(38, 38)
        self.mic_btn.setStyleSheet(self._button_style(self.colors["primary"]))
        self.mic_btn.clicked.connect(self._toggle_listening)
        input_layout.addWidget(self.mic_btn)

        main_layout.addLayout(input_layout)

        # --- Quick Actions ---
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(4)

        quick_actions = [
            ("📸", "screenshot", "Screenshot"),
            ("🔍", "search", "Search"),
            ("⚙️", "system_info", "System Info"),
            ("📁", "list files", "Files"),
        ]

        for icon, cmd, tooltip in quick_actions:
            btn = QPushButton(icon)
            btn.setFixedSize(44, 32)
            btn.setToolTip(tooltip)
            btn.setStyleSheet(self._button_style(self.colors["surface"]))
            btn.clicked.connect(lambda checked, c=cmd: self._quick_action(c))
            actions_layout.addWidget(btn)

        actions_layout.addStretch()
        main_layout.addLayout(actions_layout)

    def _button_style(self, bg: str | None = None) -> str:
        """Generate button stylesheet.

        Args:
            bg: Background color.

        Returns:
            CSS stylesheet string.
        """
        bg = bg or self.colors["surface"]
        return f"""
            QPushButton {{
                background-color: {bg};
                color: {self.colors['text']};
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.colors['primary']};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['accent']};
            }}
        """

    def _connect_signals(self) -> None:
        """Connect thread-safe signals to UI update slots."""
        self.status_signal.connect(self._update_status)
        self.response_signal.connect(self._add_conversation)
        self.listening_signal.connect(self._update_listening)

    def set_engine(self, engine: Any) -> None:
        """Connect the JARVIS engine to the UI.

        Args:
            engine: JarvisEngine instance.
        """
        self._engine = engine
        engine.set_status_callback(self.status_signal.emit)
        engine.set_response_callback(self.response_signal.emit)
        engine.set_listening_callback(self.listening_signal.emit)

    def _on_send(self) -> None:
        """Handle send button / Enter key press."""
        text = self.input_field.text().strip()
        if not text or self._engine is None:
            return

        self.input_field.clear()

        # Add user message to chat
        self._add_message(text, is_user=True)

        # Process in background thread
        import threading
        threading.Thread(
            target=self._process_input, args=(text,), daemon=True
        ).start()

    def _process_input(self, text: str) -> None:
        """Process input in background thread.

        Args:
            text: User input text.
        """
        self._engine.process_text_input(text)
        # Response will be added via the response_signal callback

    def _toggle_listening(self) -> None:
        """Toggle continuous listening mode."""
        self._is_listening = not self._is_listening
        if self._engine:
            self._engine._notify_listening(self._is_listening)

        if self._is_listening:
            self.mic_btn.setStyleSheet(self._button_style(self.colors["accent"]))
        else:
            self.mic_btn.setStyleSheet(self._button_style(self.colors["primary"]))

    def _quick_action(self, command: str) -> None:
        """Execute a quick action button command.

        Args:
            command: Command to execute.
        """
        if self._engine:
            self._add_message(command, is_user=True)
            import threading
            threading.Thread(
                target=self._process_input, args=(command,), daemon=True
            ).start()

    def _add_message(self, text: str, is_user: bool = True) -> None:
        """Add a message bubble to the chat area.

        Args:
            text: Message text.
            is_user: True for user messages.
        """
        bubble = MessageBubble(text, is_user=is_user, colors=self.colors)
        self.chat_layout.addWidget(bubble)

        # Auto-scroll to bottom
        QTimer.singleShot(100, self._scroll_to_bottom)

    def _add_conversation(self, user_input: str, response: str) -> None:
        """Add a complete conversation exchange.

        Args:
            user_input: User's message.
            response: JARVIS's response.
        """
        # Clean action tags from response for display
        display_response = response
        if "[ACTION:" in display_response:
            bracket_end = display_response.find("]")
            if bracket_end != -1:
                display_response = display_response[bracket_end + 1:].strip()

        self._add_message(display_response, is_user=False)

    def _update_status(self, status: str) -> None:
        """Update the status label.

        Args:
            status: Status message.
        """
        color = self.colors["success"] if "ready" in status.lower() else self.colors["warning"]
        self.status_label.setStyleSheet(f"""
            color: {color};
            font-size: 11px;
            padding: 4px 8px;
            background-color: {self.colors['surface']};
            border-radius: 8px;
        """)
        self.status_label.setText(status)

    def _update_listening(self, listening: bool) -> None:
        """Update the listening indicator.

        Args:
            listening: Whether JARVIS is listening.
        """
        self._is_listening = listening
        self.listening_indicator.setVisible(listening)
        if listening:
            self.mic_btn.setStyleSheet(self._button_style(self.colors["accent"]))
        else:
            self.mic_btn.setStyleSheet(self._button_style(self.colors["primary"]))

    def _scroll_to_bottom(self) -> None:
        """Scroll chat area to the bottom."""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_close(self) -> None:
        """Handle close button click."""
        if self._engine:
            self._engine.stop()
        self.close()
        QApplication.quit()

    # --- Draggable Window ---

    def mousePressEvent(self, event: Any) -> None:
        """Handle mouse press for window dragging."""
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: Any) -> None:
        """Handle mouse move for window dragging."""
        if self._dragging and self._drag_position is not None:
            self.move(event.globalPos() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event: Any) -> None:
        """Handle mouse release to stop dragging."""
        self._dragging = False
        self._drag_position = None


def create_overlay(config: dict[str, Any] | None = None) -> "OverlayWindow | None":
    """Create and return the overlay window.

    Args:
        config: UI configuration dictionary.

    Returns:
        OverlayWindow instance or None if PyQt5 is unavailable.
    """
    if not PYQT_AVAILABLE:
        logger.error("Cannot create overlay: PyQt5 not installed")
        return None

    return OverlayWindow(config)
