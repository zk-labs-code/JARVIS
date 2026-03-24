"""Coding assistant module for JARVIS.

Provides code generation, execution, formatting, and debugging
capabilities with support for multiple programming languages.
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("jarvis.actions.coding")


class CodingAssistant:
    """Handles code writing, execution, formatting, and debugging."""

    def __init__(self, max_execution_time: int = 30):
        """Initialize the coding assistant.

        Args:
            max_execution_time: Maximum execution time in seconds.
        """
        self.max_execution_time = max_execution_time
        self._temp_dir = Path(tempfile.mkdtemp(prefix="jarvis_code_"))
        self._supported_languages = {
            "python": {"ext": ".py", "cmd": [sys.executable]},
            "javascript": {"ext": ".js", "cmd": ["node"]},
            "bash": {"ext": ".sh", "cmd": ["bash"]},
            "powershell": {"ext": ".ps1", "cmd": ["powershell", "-File"]},
            "html": {"ext": ".html", "cmd": None},  # Open in browser
        }
        logger.info(f"Coding assistant initialized. Temp dir: {self._temp_dir}")

    def execute(self, code_or_command: str) -> str:
        """Execute code or process a coding command.

        Args:
            code_or_command: Code to execute or a coding task description.

        Returns:
            Execution result or generated code.
        """
        code = code_or_command.strip()

        # If it looks like actual code, run it
        if self._looks_like_code(code):
            language = self._detect_language(code)
            return self.run_code(code, language)

        # Otherwise treat as a task description
        return f"Coding task noted: {code}\nPlease provide the code to execute, or ask me to write code for a specific task."

    def run_code(self, code: str, language: str = "python") -> str:
        """Execute code in the specified language.

        Args:
            code: Source code to execute.
            language: Programming language.

        Returns:
            Execution output.
        """
        lang_info = self._supported_languages.get(language)
        if not lang_info:
            return f"Unsupported language: {language}. Supported: {', '.join(self._supported_languages)}"

        if lang_info["cmd"] is None:
            return self._handle_static_file(code, language)

        # Write code to temp file
        ext = lang_info["ext"]
        temp_file = self._temp_dir / f"script{ext}"
        temp_file.write_text(code, encoding="utf-8")

        # Execute
        cmd = lang_info["cmd"] + [str(temp_file)]

        try:
            logger.info(f"Executing {language} code...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
                cwd=str(self._temp_dir),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += f"[stderr] {result.stderr}"
            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"

            return output.strip() if output.strip() else "Code executed successfully (no output)"

        except subprocess.TimeoutExpired:
            return f"Execution timed out after {self.max_execution_time} seconds"
        except FileNotFoundError:
            return f"Runtime not found for {language}. Is it installed?"
        except Exception as e:
            return f"Execution error: {str(e)}"
        finally:
            # Clean up temp file
            try:
                temp_file.unlink(missing_ok=True)
            except OSError:
                pass

    def _handle_static_file(self, code: str, language: str) -> str:
        """Handle static files (HTML, CSS) by saving and opening.

        Args:
            code: File content.
            language: File language.

        Returns:
            Result message.
        """
        ext = self._supported_languages[language]["ext"]
        filepath = self._temp_dir / f"output{ext}"
        filepath.write_text(code, encoding="utf-8")

        # Try to open in default browser
        try:
            import webbrowser
            webbrowser.open(str(filepath))
            return f"File saved and opened: {filepath}"
        except Exception:
            return f"File saved: {filepath}"

    def format_code(self, code: str, language: str = "python") -> str:
        """Format code using appropriate formatter.

        Args:
            code: Code to format.
            language: Programming language.

        Returns:
            Formatted code.
        """
        if language == "python":
            return self._format_python(code)
        return code

    def _format_python(self, code: str) -> str:
        """Format Python code using black or autopep8.

        Args:
            code: Python code to format.

        Returns:
            Formatted code.
        """
        try:
            import black
            formatted = black.format_str(code, mode=black.Mode())
            return formatted
        except ImportError:
            pass

        try:
            import autopep8
            formatted = autopep8.fix_code(code)
            return formatted
        except ImportError:
            pass

        return code

    def analyze_code(self, code: str, language: str = "python") -> str:
        """Analyze code for potential issues.

        Args:
            code: Code to analyze.
            language: Programming language.

        Returns:
            Analysis report.
        """
        if language != "python":
            return "Code analysis currently only supports Python."

        issues = []

        # Basic static analysis
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.rstrip()

            # Check for common issues
            if len(stripped) > 120:
                issues.append(f"Line {i}: Line too long ({len(stripped)} chars)")

            if "import *" in stripped:
                issues.append(f"Line {i}: Wildcard import (import *)")

            if stripped.endswith("  "):
                issues.append(f"Line {i}: Trailing whitespace")

            if "eval(" in stripped or "exec(" in stripped:
                issues.append(f"Line {i}: Use of eval/exec (security risk)")

            if "password" in stripped.lower() and "=" in stripped:
                issues.append(f"Line {i}: Possible hardcoded password")

        # Try running with py_compile
        temp_file = self._temp_dir / "analyze_temp.py"
        temp_file.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(temp_file)],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                issues.append(f"Syntax error: {result.stderr.strip()}")
        except Exception:
            pass
        finally:
            temp_file.unlink(missing_ok=True)

        if not issues:
            return "No issues found. Code looks good!"

        report = f"Found {len(issues)} issue(s):\n"
        for issue in issues:
            report += f"  - {issue}\n"
        return report

    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like executable code.

        Args:
            text: Text to check.

        Returns:
            True if text appears to be code.
        """
        code_indicators = [
            "print(", "import ", "def ", "class ", "if ", "for ", "while ",
            "return ", "from ", "try:", "except", "with ",
            "console.log(", "function ", "const ", "let ", "var ",
            "#!/", "echo ", "export ",
            "<html", "<div", "<!DOCTYPE",
        ]
        return any(indicator in text for indicator in code_indicators)

    def _detect_language(self, code: str) -> str:
        """Detect the programming language of code.

        Args:
            code: Source code.

        Returns:
            Detected language name.
        """
        if code.startswith("#!") and ("python" in code.split("\n")[0]):
            return "python"
        if code.startswith("#!") and ("bash" in code.split("\n")[0] or "sh" in code.split("\n")[0]):
            return "bash"
        if "<html" in code.lower() or "<!doctype" in code.lower():
            return "html"

        # Python indicators
        python_indicators = [
            "import ", "from ", "def ", "class ", "print(",
            "if __name__", "self.", "elif ", "except:",
        ]
        if any(ind in code for ind in python_indicators):
            return "python"

        # JavaScript indicators
        js_indicators = [
            "console.log(", "function ", "const ", "let ", "var ",
            "=>", "require(", "module.exports",
        ]
        if any(ind in code for ind in js_indicators):
            return "javascript"

        # Bash indicators
        bash_indicators = ["echo ", "export ", "#!/bin/", "fi\n", "done\n"]
        if any(ind in code for ind in bash_indicators):
            return "bash"

        # Default to Python
        return "python"

    def create_project(self, name: str, language: str = "python") -> str:
        """Create a new project scaffold.

        Args:
            name: Project name.
            language: Programming language.

        Returns:
            Result message.
        """
        project_dir = Path.cwd() / name
        project_dir.mkdir(parents=True, exist_ok=True)

        if language == "python":
            (project_dir / "__init__.py").write_text("")
            (project_dir / "main.py").write_text(
                '"""Main module."""\n\n\ndef main():\n    print("Hello from '
                f'{name}!")\n\n\nif __name__ == "__main__":\n    main()\n'
            )
            (project_dir / "requirements.txt").write_text("")
            return f"Python project created: {project_dir}"

        return f"Project created: {project_dir}"

    def cleanup(self) -> None:
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass
