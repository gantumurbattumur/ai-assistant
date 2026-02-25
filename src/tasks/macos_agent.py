"""macOS task agent — set alarms, get calendar events, write notes.

All functions execute AppleScript via ``subprocess.run(['osascript', '-e', ...])``
and return human-readable strings.  They are safe to call on non-macOS, returning
a "not supported" message instead of crashing.

Security note
-------------
macOS will prompt the user the first time the terminal (or IDE) requests
access to Reminders, Calendar, or Notes.  The user must grant permission via:

    System Settings ▸ Privacy & Security ▸ Automation

If permission is denied, the functions return a descriptive error string.
"""
from __future__ import annotations

import platform
import subprocess
import shlex
from datetime import datetime, timedelta
from typing import Optional


# ── Helpers ─────────────────────────────────────────────────────

def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _run_applescript(script: str) -> tuple[bool, str]:
    """Run an AppleScript snippet via osascript.

    Returns (success: bool, output_or_error: str).
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            stderr = result.stderr.strip()
            # Common macOS permission error
            if "not allowed" in stderr.lower() or "assistive" in stderr.lower():
                return False, (
                    f"Permission denied. Please grant Automation access:\n"
                    f"  System Settings ▸ Privacy & Security ▸ Automation\n"
                    f"  (Details: {stderr})"
                )
            return False, f"AppleScript error: {stderr}"
    except FileNotFoundError:
        return False, "osascript not found — are you on macOS?"
    except subprocess.TimeoutExpired:
        return False, "AppleScript timed out (30 s)."
    except Exception as exc:
        return False, f"Unexpected error running AppleScript: {exc}"


def _escape(text: str) -> str:
    """Escape a string for safe embedding inside AppleScript double-quotes."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


# ── Public task functions ───────────────────────────────────────

def set_alarm(time_str: str, title: str = "AI Assistant Alarm") -> str:
    """Create a reminder in macOS Reminders app at the specified time.

    Parameters
    ----------
    time_str : str
        Natural-ish time string.  Accepts:
        - Relative: "5 minutes", "1 hour", "30 min"
        - Absolute: "14:30", "2:30 PM", "2026-02-25 14:30"
    title : str
        Reminder title.

    Returns
    -------
    str  Human-readable success or error message.
    """
    if not _is_macos():
        return "⚠️  Setting alarms is only supported on macOS."

    # ── Parse time ──────────────────────────────────────────────
    remind_at: Optional[datetime] = None

    # Relative: "5 minutes", "1 hour", "30 min"
    import re
    rel = re.match(r"(\d+)\s*(min(?:ute)?s?|hours?|hrs?|secs?|seconds?)", time_str, re.I)
    if rel:
        amount = int(rel.group(1))
        unit = rel.group(2).lower()
        if unit.startswith("min"):
            delta = timedelta(minutes=amount)
        elif unit.startswith("h"):
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(seconds=amount)
        remind_at = datetime.now() + delta

    # Absolute HH:MM or H:MM AM/PM (today)
    if remind_at is None:
        for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
            try:
                t = datetime.strptime(time_str.strip(), fmt)
                now = datetime.now()
                remind_at = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
                if remind_at < now:
                    remind_at += timedelta(days=1)  # next occurrence
                break
            except ValueError:
                continue

    # Full datetime string
    if remind_at is None:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M"):
            try:
                remind_at = datetime.strptime(time_str.strip(), fmt)
                break
            except ValueError:
                continue

    if remind_at is None:
        return (
            f"❌ Could not parse time '{time_str}'.  "
            "Try formats like '5 minutes', '14:30', '2:30 PM', or '2026-02-25 14:30'."
        )

    # AppleScript date format: "month day, year at HH:MM:SS"
    as_date = remind_at.strftime("%B %d, %Y at %H:%M:%S")
    safe_title = _escape(title)

    script = f'''
    tell application "Reminders"
        set newReminder to make new reminder with properties {{name:"{safe_title}", due date:date "{as_date}"}}
    end tell
    '''

    ok, msg = _run_applescript(script)
    if ok:
        friendly = remind_at.strftime("%Y-%m-%d %I:%M %p")
        return f"✅ Alarm set: \"{title}\" at {friendly}"
    return f"❌ Failed to set alarm: {msg}"


def get_calendar_events(date_str: str = "today") -> str:
    """Fetch calendar events for a given date from macOS Calendar.

    Parameters
    ----------
    date_str : str
        "today", "tomorrow", "YYYY-MM-DD", or "MM/DD/YYYY".

    Returns
    -------
    str  Formatted list of events, or an error message.
    """
    if not _is_macos():
        return "⚠️  Getting calendar events is only supported on macOS."

    # ── Resolve date ────────────────────────────────────────────
    target: Optional[datetime] = None
    low = date_str.strip().lower()

    if low in ("today", ""):
        target = datetime.now()
    elif low == "tomorrow":
        target = datetime.now() + timedelta(days=1)
    elif low == "yesterday":
        target = datetime.now() - timedelta(days=1)
    else:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y"):
            try:
                target = datetime.strptime(date_str.strip(), fmt)
                break
            except ValueError:
                continue

    if target is None:
        return f"❌ Could not parse date '{date_str}'. Try 'today', 'tomorrow', or 'YYYY-MM-DD'."

    start_date = target.strftime("%B %d, %Y") + " at 00:00:00"
    end_date = target.strftime("%B %d, %Y") + " at 23:59:59"
    display_date = target.strftime("%A, %B %d, %Y")

    script = f'''
    set output to ""
    tell application "Calendar"
        set startDate to date "{start_date}"
        set endDate to date "{end_date}"
        repeat with c in calendars
            set evts to (every event of c whose start date ≥ startDate and start date ≤ endDate)
            repeat with e in evts
                set evtName to summary of e
                set evtStart to start date of e
                set evtEnd to end date of e
                set output to output & evtName & " | " & (evtStart as string) & " — " & (evtEnd as string) & linefeed
            end repeat
        end repeat
    end tell
    return output
    '''

    ok, msg = _run_applescript(script)
    if ok:
        if msg.strip() == "":
            return f"📅 No events found for {display_date}."
        header = f"📅 Events for {display_date}:\n"
        lines = [f"  • {line}" for line in msg.strip().splitlines() if line.strip()]
        return header + "\n".join(lines)
    return f"❌ Failed to get calendar events: {msg}"


def write_note(content: str, title: str = "") -> str:
    """Create a new note in macOS Notes app.

    Parameters
    ----------
    content : str
        Body text. HTML is accepted by Notes.app.
    title : str
        Note title. If empty, the first line of content is used.

    Returns
    -------
    str  Success or error message.
    """
    if not _is_macos():
        return "⚠️  Writing notes is only supported on macOS."

    if not content.strip():
        return "❌ Cannot create an empty note."

    if not title:
        first_line = content.strip().splitlines()[0][:80]
        title = first_line

    safe_title = _escape(title)
    safe_body = _escape(content)

    script = f'''
    tell application "Notes"
        tell account "iCloud"
            make new note at folder "Notes" with properties {{name:"{safe_title}", body:"{safe_body}"}}
        end tell
    end tell
    '''

    ok, msg = _run_applescript(script)
    if ok:
        return f"📝 Note created: \"{title}\""
    # If iCloud account fails, try without specifying account
    script_fallback = f'''
    tell application "Notes"
        make new note with properties {{name:"{safe_title}", body:"{safe_body}"}}
    end tell
    '''
    ok2, msg2 = _run_applescript(script_fallback)
    if ok2:
        return f"📝 Note created: \"{title}\""
    return f"❌ Failed to create note: {msg2}"


# ── Task agent node (for the multi-agent graph) ────────────────

def task_agent_node(state: dict) -> dict:
    """Multi-agent graph node — detects and executes macOS tasks.

    The coordinator places 'task_agent' in the plan. This node inspects
    the query to determine *which* task to run, executes it, and appends
    the result to agent_results.
    """
    from src.agents import AgentResult

    query: str = state.get("translated_query") or state["query"]
    q_lower = query.lower()

    result_text = ""
    sources: list[str] = []

    # ── Dispatch to the right task ──────────────────────────────
    if _match_alarm(q_lower):
        time_str, title = _extract_alarm_params(query)
        result_text = set_alarm(time_str, title)
        sources = ["macOS Reminders"]

    elif _match_calendar(q_lower):
        date_str = _extract_date_param(query)
        result_text = get_calendar_events(date_str)
        sources = ["macOS Calendar"]

    elif _match_note(q_lower):
        note_content, note_title = _extract_note_params(query)
        result_text = write_note(note_content, note_title)
        sources = ["macOS Notes"]

    else:
        result_text = (
            "I detected a task intent but couldn't determine the specific action. "
            "Supported tasks: set alarm/reminder, get calendar events, write a note."
        )

    agent_result: AgentResult = {
        "agent": "task_agent",
        "content": result_text,
        "sources": sources,
        "confidence": "high" if result_text.startswith(("✅", "📅", "📝")) else "low",
    }

    prior = state.get("agent_results", [])
    used = state.get("agents_used", [])

    return {
        "agent_results": prior + [agent_result],
        "agents_used": used + ["🖥️  Task Agent (macOS)"],
    }


# ── Intent matchers (used by both coordinator and task_agent) ──

def _match_alarm(text: str) -> bool:
    import re
    return bool(re.search(
        r"(set\s+(an?\s+)?alarm|set\s+(an?\s+)?reminder|remind\s+me|wake\s+me)", text, re.I
    ))

def _match_calendar(text: str) -> bool:
    import re
    return bool(re.search(
        r"(calendar|schedule|events?\s+(for|on|today|tomorrow)|my\s+events|what('s| is)\s+on\s+my)", text, re.I
    ))

def _match_note(text: str) -> bool:
    import re
    return bool(re.search(
        r"(write\s+(a\s+)?note|create\s+(a\s+)?note|save\s+(a\s+)?note|take\s+(a\s+)?note|note\s+down)", text, re.I
    ))

def matches_any_task(text: str) -> bool:
    """Return True if the text matches any known macOS task intent.

    Used by the coordinator to detect task queries.
    """
    low = text.lower()
    return _match_alarm(low) or _match_calendar(low) or _match_note(low)


# ── Parameter extraction helpers ────────────────────────────────

def _extract_alarm_params(query: str) -> tuple[str, str]:
    """Best-effort extraction of time and title from a natural query."""
    import re

    # Try to find "at HH:MM" or "in N minutes/hours"
    time_str = ""
    title = "AI Assistant Alarm"

    # "in 5 minutes", "in 1 hour"
    m = re.search(r"in\s+(\d+\s*(?:min(?:ute)?s?|hours?|hrs?))", query, re.I)
    if m:
        time_str = m.group(1)
    else:
        # "at 14:30", "at 2:30 PM"
        m = re.search(r"at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)", query, re.I)
        if m:
            time_str = m.group(1)
        else:
            # "for 2026-02-25 14:30"
            m = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2})", query)
            if m:
                time_str = m.group(1)
            else:
                # fallback: grab any time-like pattern
                m = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)", query)
                if m:
                    time_str = m.group(1)
                else:
                    time_str = "5 minutes"  # safe default

    # Title: "to <title>", "called <title>", "named <title>"
    m = re.search(r"(?:to|called|named|titled|for)\s+['\"]?(.+?)['\"]?\s*$", query, re.I)
    if m:
        candidate = m.group(1).strip().rstrip(".")
        if len(candidate) > 2 and not any(
            kw in candidate.lower() for kw in ("minute", "hour", "am", "pm", ":")
        ):
            title = candidate

    return time_str, title


def _extract_date_param(query: str) -> str:
    """Extract a date from a calendar query."""
    import re

    low = query.lower()
    if "tomorrow" in low:
        return "tomorrow"
    if "yesterday" in low:
        return "yesterday"

    # YYYY-MM-DD
    m = re.search(r"(\d{4}-\d{2}-\d{2})", query)
    if m:
        return m.group(1)

    # MM/DD/YYYY
    m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", query)
    if m:
        return m.group(1)

    return "today"


def _extract_note_params(query: str) -> tuple[str, str]:
    """Extract note content and optional title."""
    import re

    title = ""
    content = query

    # "write a note: <content>" / "note down: <content>"
    m = re.search(r"(?:note|write\s+a\s+note|save\s+a\s+note|take\s+a\s+note)[:\s]+(.+)", query, re.I | re.S)
    if m:
        content = m.group(1).strip()

    # "titled <title>: <content>"
    m = re.search(r"titled?\s+['\"]?(.+?)['\"]?\s*:\s*(.+)", content, re.I | re.S)
    if m:
        title = m.group(1).strip()
        content = m.group(2).strip()

    return content, title
