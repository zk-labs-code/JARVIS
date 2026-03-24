"""System control module for JARVIS.

Provides system-level operations like launching apps, managing processes,
taking screenshots, controlling volume, and file operations.
"""

import datetime
import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("jarvis.actions.system")


class SystemController:
    """Handles system-level operations and computer control."""

    def __init__(self):
        """Initialize the system controller."""
        self.platform = platform.system().lower()
        self._is_windows = self.platform == "windows"
        self._is_linux = self.platform == "linux"
        self._is_mac = self.platform == "darwin"
        logger.info(f"System controller initialized for {platform.system()}")

    def execute(self, command: str) -> str:
        """Execute a system command by parsing the action string.

        Args:
            command: Command string to execute.

        Returns:
            Result string.
        """
        command = command.strip().lower()

        if command.startswith("open "):
            return self.open_application(command[5:].strip())
        elif command.startswith("close "):
            return self.close_application(command[6:].strip())
        elif command == "screenshot":
            return self.take_screenshot()
        elif command in ("get_time", "time"):
            return self.get_time()
        elif command in ("get_date", "date"):
            return self.get_date()
        elif command.startswith("volume"):
            return self._handle_volume(command)
        elif command in ("scroll_up", "scroll_down"):
            return self._handle_scroll(command)
        elif command == "lock":
            return self.lock_screen()
        elif command.startswith("type "):
            return self.type_text(command[5:])
        elif command == "shutdown":
            return "Shutdown requires explicit confirmation. Use 'confirm shutdown' to proceed."
        elif command == "restart":
            return "Restart requires explicit confirmation. Use 'confirm restart' to proceed."
        else:
            return self._run_shell_command(command)

    def open_application(self, app_name: str) -> str:
        """Open an application by name.

        Args:
            app_name: Name of the application to open.

        Returns:
            Result message.
        """
        app_name = app_name.strip().lower()
        logger.info(f"Opening application: {app_name}")

        # Common application mappings
        app_map_windows = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "paint": "mspaint.exe",
            "explorer": "explorer.exe",
            "file explorer": "explorer.exe",
            "task manager": "taskmgr.exe",
            "command prompt": "cmd.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "terminal": "wt.exe",
            "settings": "ms-settings:",
            "control panel": "control.exe",
            "browser": "start https://www.google.com",
            "chrome": "chrome.exe",
            "firefox": "firefox.exe",
            "edge": "msedge.exe",
            "word": "winword.exe",
            "excel": "excel.exe",
            "powerpoint": "powerpnt.exe",
            "vscode": "code",
            "vs code": "code",
        }

        app_map_linux = {
            "terminal": "x-terminal-emulator",
            "browser": "xdg-open https://www.google.com",
            "file manager": "nautilus",
            "files": "nautilus",
            "calculator": "gnome-calculator",
            "settings": "gnome-control-center",
            "text editor": "gedit",
            "vscode": "code",
            "vs code": "code",
            "firefox": "firefox",
            "chrome": "google-chrome",
        }

        app_map = app_map_windows if self._is_windows else app_map_linux
        executable = app_map.get(app_name, app_name)

        try:
            if self._is_windows:
                if executable.startswith("ms-settings:"):
                    os.startfile(executable)
                elif executable.startswith("start "):
                    subprocess.Popen(executable, shell=True)
                else:
                    subprocess.Popen(executable, shell=True)
            else:
                subprocess.Popen(
                    executable.split(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return f"Opened {app_name}"

        except FileNotFoundError:
            return f"Application not found: {app_name}"
        except Exception as e:
            return f"Failed to open {app_name}: {str(e)}"

    def close_application(self, app_name: str) -> str:
        """Close an application by name.

        Args:
            app_name: Name of the application to close.

        Returns:
            Result message.
        """
        logger.info(f"Closing application: {app_name}")
        try:
            if self._is_windows:
                subprocess.run(
                    ["taskkill", "/IM", f"{app_name}.exe", "/F"],
                    capture_output=True, text=True, timeout=10,
                )
            else:
                subprocess.run(
                    ["pkill", "-f", app_name],
                    capture_output=True, text=True, timeout=10,
                )
            return f"Closed {app_name}"
        except Exception as e:
            return f"Failed to close {app_name}: {str(e)}"

    def take_screenshot(self) -> str:
        """Take a screenshot and save it.

        Returns:
            Path to the saved screenshot.
        """
        try:
            import pyautogui

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshots_dir = Path("data/screenshots")
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            filepath = screenshots_dir / f"screenshot_{timestamp}.png"

            screenshot = pyautogui.screenshot()
            screenshot.save(str(filepath))
            logger.info(f"Screenshot saved: {filepath}")
            return f"Screenshot saved to {filepath}"

        except ImportError:
            return "PyAutoGUI not available for screenshots"
        except Exception as e:
            return f"Screenshot failed: {str(e)}"

    def get_time(self) -> str:
        """Get the current time.

        Returns:
            Current time string.
        """
        now = datetime.datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}"

    def get_date(self) -> str:
        """Get the current date.

        Returns:
            Current date string.
        """
        now = datetime.datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}"

    def _handle_volume(self, command: str) -> str:
        """Handle volume control commands.

        Args:
            command: Volume command string.

        Returns:
            Result message.
        """
        try:
            if "mute" in command:
                if self._is_windows:
                    from ctypes import POINTER, cast

                    from comtypes import CLSCTX_ALL
                    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                    devices = AudioUtilities.GetSpeakers()
                    interface = devices.Activate(
                        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                    volume = cast(interface, POINTER(IAudioEndpointVolume))
                    volume.SetMute(1, None)
                return "Volume muted"

            elif "up" in command or "increase" in command:
                if self._is_windows:
                    subprocess.run(
                        ["powershell", "-c",
                         "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"],
                        capture_output=True, timeout=5,
                    )
                elif self._is_linux:
                    subprocess.run(
                        ["amixer", "-D", "pulse", "sset", "Master", "10%+"],
                        capture_output=True, timeout=5,
                    )
                return "Volume increased"

            elif "down" in command or "decrease" in command:
                if self._is_windows:
                    subprocess.run(
                        ["powershell", "-c",
                         "(New-Object -ComObject WScript.Shell).SendKeys([char]174)"],
                        capture_output=True, timeout=5,
                    )
                elif self._is_linux:
                    subprocess.run(
                        ["amixer", "-D", "pulse", "sset", "Master", "10%-"],
                        capture_output=True, timeout=5,
                    )
                return "Volume decreased"

            return "Volume command not recognized. Use: volume up, volume down, or volume mute"

        except Exception as e:
            return f"Volume control error: {str(e)}"

    def _handle_scroll(self, command: str) -> str:
        """Handle scroll commands.

        Args:
            command: Scroll direction.

        Returns:
            Result message.
        """
        try:
            import pyautogui

            if "up" in command:
                pyautogui.scroll(3)
                return "Scrolled up"
            else:
                pyautogui.scroll(-3)
                return "Scrolled down"
        except ImportError:
            return "PyAutoGUI not available for scrolling"
        except Exception as e:
            return f"Scroll error: {str(e)}"

    def lock_screen(self) -> str:
        """Lock the computer screen.

        Returns:
            Result message.
        """
        try:
            if self._is_windows:
                subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], timeout=5)
            elif self._is_linux:
                subprocess.run(["loginctl", "lock-session"], timeout=5)
            elif self._is_mac:
                subprocess.run(
                    ["osascript", "-e",
                     'tell application "System Events" to keystroke "q" '
                     'using {command down, control down}'],
                    timeout=5,
                )
            return "Screen locked"
        except Exception as e:
            return f"Failed to lock screen: {str(e)}"

    def type_text(self, text: str) -> str:
        """Type text using keyboard simulation.

        Args:
            text: Text to type.

        Returns:
            Result message.
        """
        try:
            import pyautogui
            pyautogui.typewrite(text, interval=0.02)
            return f"Typed: {text}"
        except ImportError:
            return "PyAutoGUI not available for typing"
        except Exception as e:
            return f"Typing error: {str(e)}"

    def file_operation(self, command: str) -> str:
        """Handle file system operations.

        Args:
            command: File operation command.

        Returns:
            Result message.
        """
        command = command.strip().lower()

        try:
            if "create file" in command:
                parts = command.replace("create file", "").strip().split(maxsplit=1)
                filepath = parts[0] if parts else "new_file.txt"
                content = parts[1] if len(parts) > 1 else ""
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                Path(filepath).write_text(content)
                return f"Created file: {filepath}"

            elif "delete file" in command:
                filepath = command.replace("delete file", "").strip()
                if Path(filepath).exists():
                    Path(filepath).unlink()
                    return f"Deleted: {filepath}"
                return f"File not found: {filepath}"

            elif "list files" in command:
                directory = command.replace("list files", "").strip() or "."
                if Path(directory).is_dir():
                    files = list(Path(directory).iterdir())
                    file_list = "\n".join(
                        f"  {'[DIR]' if f.is_dir() else '[FILE]'} {f.name}"
                        for f in sorted(files)
                    )
                    return f"Contents of {directory}:\n{file_list}"
                return f"Not a directory: {directory}"

            elif "copy file" in command:
                parts = command.replace("copy file", "").strip().split(" to ")
                if len(parts) == 2:
                    shutil.copy2(parts[0].strip(), parts[1].strip())
                    return f"Copied {parts[0].strip()} to {parts[1].strip()}"
                return "Usage: copy file <source> to <destination>"

            elif "move file" in command or "rename file" in command:
                keyword = "move file" if "move file" in command else "rename file"
                parts = command.replace(keyword, "").strip().split(" to ")
                if len(parts) == 2:
                    shutil.move(parts[0].strip(), parts[1].strip())
                    return f"Moved {parts[0].strip()} to {parts[1].strip()}"
                return "Usage: move file <source> to <destination>"

            else:
                return f"Unknown file operation: {command}"

        except PermissionError:
            return f"Permission denied for: {command}"
        except Exception as e:
            return f"File operation error: {str(e)}"

    def app_control(self, command: str) -> str:
        """Control running applications (minimize, maximize, focus).

        Args:
            command: App control command.

        Returns:
            Result message.
        """
        try:
            import pyautogui

            if "minimize" in command:
                if self._is_windows:
                    pyautogui.hotkey("win", "d")
                return "Windows minimized"
            elif "maximize" in command:
                if self._is_windows:
                    pyautogui.hotkey("win", "up")
                return "Window maximized"
            elif "switch" in command or "alt tab" in command:
                pyautogui.hotkey("alt", "tab")
                return "Switched window"
            else:
                return f"Unknown app control: {command}"

        except ImportError:
            return "PyAutoGUI not available"
        except Exception as e:
            return f"App control error: {str(e)}"

    def _run_shell_command(self, command: str) -> str:
        """Run a shell command and return output.

        Args:
            command: Shell command to run.

        Returns:
            Command output.
        """
        # Safety check: block dangerous commands
        dangerous = ["rm -rf /", "format c:", "del /f /s /q", "mkfs", ":(){:|:&};:"]
        for d in dangerous:
            if d in command:
                return f"Blocked dangerous command: {command}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout.strip()
            if result.returncode != 0 and result.stderr:
                output += f"\nError: {result.stderr.strip()}"
            return output if output else "Command executed successfully (no output)"

        except subprocess.TimeoutExpired:
            return f"Command timed out: {command}"
        except Exception as e:
            return f"Command execution error: {str(e)}"

    def get_system_info(self) -> str:
        """Get system information summary.

        Returns:
            System info string.
        """
        import psutil

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        info = (
            f"System: {platform.system()} {platform.release()}\n"
            f"CPU: {platform.processor()} ({psutil.cpu_count()} cores) - {cpu_percent}% usage\n"
            f"RAM: {memory.used / (1024**3):.1f}/{memory.total / (1024**3):.1f} GB "
            f"({memory.percent}%)\n"
            f"Disk: {disk.used / (1024**3):.1f}/{disk.total / (1024**3):.1f} GB "
            f"({disk.percent}%)\n"
            f"Python: {platform.python_version()}"
        )
        return info
