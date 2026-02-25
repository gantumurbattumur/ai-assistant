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

_P = platform.system()


def set_alarm(time_str: str, title: str = "AI Assistant Alarm") -> str:
    return f"⚠️  Setting alarms is not yet supported on {_P}."


def get_calendar_events(date_str: str = "today") -> str:
    return f"⚠️  Calendar integration is not yet supported on {_P}."


def create_calendar_event(title: str, start_time: str, **kwargs) -> str:
    return f"⚠️  Creating calendar events is not yet supported on {_P}."


def write_note(content: str, title: str = "") -> str:
    return f"⚠️  Notes integration is not yet supported on {_P}."


def run_timer(duration_str: str) -> str:
    # Timer is pure Python — works everywhere.  Import from macos_agent.
    from src.tasks.macos_agent import run_timer as _run
    return _run(duration_str)


def run_stopwatch() -> str:
    from src.tasks.macos_agent import run_stopwatch as _run
    return _run()


def world_clock(city_or_tz: str) -> str:
    from src.tasks.macos_agent import world_clock as _wc
    return _wc(city_or_tz)


def get_task_runner():
    """Return the platform-appropriate task module.

    On macOS → ``src.tasks.macos_agent``
    Otherwise → this module (stubs + pure-Python features).
    """
    if platform.system() == "Darwin":
        from src.tasks import macos_agent
        return macos_agent
    return __import__(__name__)
