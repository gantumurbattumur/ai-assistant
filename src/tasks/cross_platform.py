"""Cross-platform task stubs.

This module provides the same interface as ``macos_agent`` but for
non-macOS platforms.  Currently all functions return "not supported"
messages.  Future implementations could use:

- **Linux**: ``notify-send``, ``at``, ``gnome-calendar`` D-Bus, etc.
- **Windows**: ``schtasks``, COM automation for Outlook/OneNote, etc.

The ``get_task_runner()`` factory selects the right backend at runtime.
"""
from __future__ import annotations

import platform


def set_alarm(time_str: str, title: str = "AI Assistant Alarm") -> str:
    return f"⚠️  Setting alarms is not yet supported on {platform.system()}."


def get_calendar_events(date_str: str = "today") -> str:
    return f"⚠️  Calendar integration is not yet supported on {platform.system()}."


def write_note(content: str, title: str = "") -> str:
    return f"⚠️  Notes integration is not yet supported on {platform.system()}."


def get_task_runner():
    """Return the platform-appropriate task module.

    On macOS → ``src.tasks.macos_agent``
    Otherwise → this module (stubs).
    """
    if platform.system() == "Darwin":
        from src.tasks import macos_agent
        return macos_agent
    return __import__(__name__)
