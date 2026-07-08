"""Native folder picker helpers for local-agent interactive jobs."""

from __future__ import annotations

from pathlib import Path
import platform
import subprocess

DEFAULT_FOLDER_PICKER_TIMEOUT_SECONDS = 150


class FolderPickerCancelled(RuntimeError):
    """Raised when the user dismisses the native folder picker."""


class FolderPickerUnavailable(RuntimeError):
    """Raised when the current platform cannot show a native folder picker."""


def pick_folder(
    *,
    title: str | None = None,
    initial_directory: str | None = None,
    timeout_seconds: float = DEFAULT_FOLDER_PICKER_TIMEOUT_SECONDS,
) -> str:
    """Open a native folder picker and return the selected absolute path."""
    if platform.system() != "Darwin":
        raise FolderPickerUnavailable("folder picker is only available on macOS daemons; type the folder path instead")
    return _pick_folder_macos(title=title, initial_directory=initial_directory, timeout_seconds=timeout_seconds)


def _pick_folder_macos(
    *,
    title: str | None,
    initial_directory: str | None,
    timeout_seconds: float,
) -> str:
    prompt = _applescript_string(title or "Choose folder to sync")
    script_lines = []
    if initial_directory:
        initial = Path(initial_directory).expanduser()
        if initial.exists():
            script_lines.append(f"set defaultLocation to POSIX file {_applescript_string(str(initial))}")
            script_lines.append(f"set selectedFolder to choose folder with prompt {prompt} default location defaultLocation")
        else:
            script_lines.append(f"set selectedFolder to choose folder with prompt {prompt}")
    else:
        script_lines.append(f"set selectedFolder to choose folder with prompt {prompt}")
    script_lines.append("POSIX path of selectedFolder")

    try:
        completed = subprocess.run(
            ["osascript", "-e", "\n".join(script_lines)],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise FolderPickerUnavailable("osascript is not available on this machine") from exc
    except subprocess.TimeoutExpired as exc:
        raise FolderPickerCancelled("folder selection timed out") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if completed.returncode == 1 and ("User canceled" in stderr or "-128" in stderr):
            raise FolderPickerCancelled("folder selection cancelled")
        raise FolderPickerUnavailable(stderr or "folder picker failed")

    selected = completed.stdout.strip()
    if not selected:
        raise FolderPickerCancelled("folder selection cancelled")
    path = Path(selected).expanduser()
    if not path.is_dir():
        raise FolderPickerUnavailable(f"selected path is not a folder: {path}")
    return str(path)


def _applescript_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
