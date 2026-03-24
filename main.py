"""JARVIS - AI Personal Assistant

Main entry point for the JARVIS AI assistant.
Initializes all subsystems and launches the overlay UI.

Usage:
    python main.py              # Launch with GUI overlay
    python main.py --no-gui     # Launch in terminal-only mode
    python main.py --config path/to/config.yaml  # Custom config
"""

import argparse
import signal
import sys
import threading

from jarvis.core.config import Config
from jarvis.core.engine import JarvisEngine
from jarvis.utils.logger import init_logging, log_separator, log_system_info


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="JARVIS - AI Personal Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in terminal-only mode without GUI overlay",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice recognition",
    )
    parser.add_argument(
        "--no-gesture",
        action="store_true",
        help="Disable gesture recognition",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def run_terminal_mode(engine: JarvisEngine) -> None:
    """Run JARVIS in terminal-only mode.

    Args:
        engine: Initialized JarvisEngine instance.
    """
    print("\n" + "=" * 50)
    print("  JARVIS - AI Personal Assistant (Terminal Mode)")
    print("=" * 50)
    print("Type commands or questions. Type 'quit' to exit.\n")

    # Start engine
    engine.start()

    while engine.is_running:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "bye", "shutdown"):
                print("\nJARVIS: Goodbye!")
                engine.stop()
                break

            if user_input.lower() == "status":
                info = engine.get_system_info()
                print("\nSystem Status:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
                continue

            if user_input.lower() == "help":
                print_help()
                continue

            # Process the command
            response = engine.process_text_input(user_input)

            # Clean response for display
            if "[ACTION:" in response:
                bracket_end = response.find("]")
                if bracket_end != -1:
                    response = response[bracket_end + 1:].strip()

            print(f"\nJARVIS: {response}")

        except KeyboardInterrupt:
            print("\n\nJARVIS: Goodbye!")
            engine.stop()
            break
        except EOFError:
            engine.stop()
            break


def print_help() -> None:
    """Print help information."""
    help_text = """
JARVIS Commands:
  General:
    help          - Show this help message
    status        - Show system status
    quit/exit     - Shut down JARVIS

  System:
    open <app>    - Open an application
    close <app>   - Close an application
    screenshot    - Take a screenshot
    time/date     - Get current time/date
    volume up/down/mute - Control volume
    lock          - Lock the screen

  Coding:
    Write any code and JARVIS will detect the language and run it.
    Example: "print('Hello World')"

  Web:
    search <query> - Search the internet
    learn <topic>  - Learn about a topic from the web

  Files:
    create file <path> - Create a new file
    list files <dir>   - List files in directory
    delete file <path> - Delete a file

  Gestures (when camera is enabled):
    Open Palm    - Activate listening
    Closed Fist  - Stop listening
    Thumbs Up    - Confirm action
    Peace Sign   - Take screenshot
    Point Up     - Scroll up
"""
    print(help_text)


def run_gui_mode(engine: JarvisEngine, config: Config) -> None:
    """Run JARVIS with the overlay GUI.

    Args:
        engine: Initialized JarvisEngine instance.
        config: Configuration instance.
    """
    try:
        from PyQt5.QtWidgets import QApplication

        from jarvis.ui.overlay import create_overlay
    except ImportError:
        print("PyQt5 not available. Falling back to terminal mode.")
        print("Install PyQt5 with: pip install PyQt5")
        run_terminal_mode(engine)
        return

    app = QApplication(sys.argv)
    app.setApplicationName("JARVIS")
    app.setQuitOnLastWindowClosed(True)

    # Create overlay window
    ui_config = config.get("ui", default={})
    window = create_overlay(ui_config)

    if window is None:
        print("Failed to create overlay. Falling back to terminal mode.")
        run_terminal_mode(engine)
        return

    # Connect engine to UI
    window.set_engine(engine)

    # Start engine in background thread
    engine_thread = threading.Thread(target=engine.start, daemon=True)
    engine_thread.start()

    # Show overlay
    window.show()

    # Run Qt event loop
    sys.exit(app.exec_())


def main() -> None:
    """Main entry point for JARVIS."""
    args = parse_args()

    # Initialize logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = init_logging(log_dir="logs", level=log_level)

    log_separator(logger, "JARVIS Starting")
    log_system_info(logger)

    # Load configuration
    config = Config(args.config)

    # Apply CLI overrides
    if args.no_voice:
        config.set(False, "voice", "listener", "enabled")
    if args.no_gesture:
        config.set(False, "gesture", "enabled")

    # Initialize engine
    engine = JarvisEngine(config)

    # Handle graceful shutdown
    def signal_handler(sig: int, frame: object) -> None:
        logger.info("Shutdown signal received")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Launch appropriate mode
    if args.no_gui:
        run_terminal_mode(engine)
    else:
        run_gui_mode(engine, config)


if __name__ == "__main__":
    main()
