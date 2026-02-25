"""macOS task agent — reminders, calendar, notes, timer, stopwatch, world clock.

All macOS-native functions execute AppleScript via
``subprocess.run(['osascript', '-e', ...])`` and return human-readable strings.
Non-AppleScript features (timer, stopwatch, world clock) are pure Python.

Security note
-------------
macOS will prompt the user the first time the terminal requests access to
Reminders, Calendar, or Notes.  Grant permission via:

    System Settings ▸ Privacy & Security ▸ Automation

If permission is denied the functions return a descriptive error string.
"""
from __future__ import annotations

import platform
import re
import subprocess
import time as _time
from datetime import datetime, timedelta
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _run_applescript(script: str) -> tuple[bool, str]:
    """Run an AppleScript snippet via osascript.

    Returns ``(success, output_or_error)``.
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
        stderr = result.stderr.strip()
        if "not allowed" in stderr.lower() or "assistive" in stderr.lower():
            return False, (
                "Permission denied. Grant Automation access:\n"
                "  System Settings ▸ Privacy & Security ▸ Automation\n"
                f"  ({stderr})"
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


def _parse_time(time_str: str) -> Optional[datetime]:
    """Parse a natural-ish time string into a datetime.

    Accepts relative ("5 minutes", "1 hour 30 min"), absolute ("14:30",
    "2:30 PM"), and full datetime ("2026-02-25 14:30").
    """
    # ── Relative: "5 minutes", "1 hour 30 min", "90 seconds" ───
    rel = re.match(
        r"(\d+)\s*(min(?:ute)?s?|hours?|hrs?|secs?|seconds?)", time_str, re.I
    )
    if rel:
        amount = int(rel.group(1))
        unit = rel.group(2).lower()
        if unit.startswith("min"):
            return datetime.now() + timedelta(minutes=amount)
        if unit.startswith("h"):
            return datetime.now() + timedelta(hours=amount)
        return datetime.now() + timedelta(seconds=amount)

    # ── Absolute HH:MM / H:MM AM|PM (today, rolls to tomorrow) ─
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
        try:
            t = datetime.strptime(time_str.strip(), fmt)
            now = datetime.now()
            dt = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            return dt if dt >= now else dt + timedelta(days=1)
        except ValueError:
            continue

    # ── Full datetime string ────────────────────────────────────
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(time_str.strip(), fmt)
        except ValueError:
            continue

    return None


def _parse_date(date_str: str) -> Optional[datetime]:
    """Resolve 'today' / 'tomorrow' / 'yesterday' / ISO / US date."""
    low = date_str.strip().lower()
    if low in ("today", ""):
        return datetime.now()
    if low == "tomorrow":
        return datetime.now() + timedelta(days=1)
    if low == "yesterday":
        return datetime.now() - timedelta(days=1)
    # "next week" → +7 days
    if low in ("next week",):
        return datetime.now() + timedelta(days=7)
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _parse_seconds(time_str: str) -> Optional[int]:
    """Parse a duration string into total seconds for timers."""
    m = re.match(
        r"(\d+)\s*(s(?:ec(?:ond)?s?)?|m(?:in(?:ute)?s?)?|h(?:(?:ou)?rs?)?)",
        time_str.strip(), re.I,
    )
    if m:
        n = int(m.group(1))
        u = m.group(2)[0].lower()
        if u == "h":
            return n * 3600
        if u == "m":
            return n * 60
        return n
    # bare number → treat as seconds
    if time_str.strip().isdigit():
        return int(time_str.strip())
    return None


# ═══════════════════════════════════════════════════════════════════
#  1.  Reminders / Alarms
# ═══════════════════════════════════════════════════════════════════

def set_alarm(time_str: str, title: str = "AI Assistant Alarm") -> str:
    """Create a timed reminder in macOS Reminders with the correct due date."""
    if not _is_macos():
        return "⚠️  Setting alarms is only supported on macOS."

    remind_at = _parse_time(time_str)
    if remind_at is None:
        return (
            f"❌ Could not parse time '{time_str}'.  "
            "Try: '5 minutes', '14:30', '2:30 PM', or '2026-02-25 14:30'."
        )

    as_date = remind_at.strftime("%B %d, %Y at %H:%M:%S")
    safe_title = _escape(title)

    script = (
        'tell application "Reminders"\n'
        f'  make new reminder with properties '
        f'{{name:"{safe_title}", due date:date "{as_date}"}}\n'
        'end tell'
    )

    ok, msg = _run_applescript(script)
    if ok:
        friendly = remind_at.strftime("%Y-%m-%d %I:%M %p")
        return f'✅ Reminder set: "{title}" at {friendly}'
    return f"❌ Failed to set alarm: {msg}"


# ═══════════════════════════════════════════════════════════════════
#  2.  Calendar — retrieve AND create events
# ═══════════════════════════════════════════════════════════════════

def get_calendar_events(date_str: str = "today") -> str:
    """Fetch calendar events for a given date from macOS Calendar."""
    if not _is_macos():
        return "⚠️  Getting calendar events is only supported on macOS."

    target = _parse_date(date_str)
    if target is None:
        return f"❌ Could not parse date '{date_str}'. Try 'today', 'tomorrow', or 'YYYY-MM-DD'."

    start_date = target.strftime("%B %d, %Y") + " at 00:00:00"
    end_date = target.strftime("%B %d, %Y") + " at 23:59:59"
    display_date = target.strftime("%A, %B %d, %Y")

    # Using ≥ / ≤ directly — osascript handles UTF-8 fine on macOS
    script = (
        'set output to ""\n'
        'tell application "Calendar"\n'
        f'  set startDate to date "{start_date}"\n'
        f'  set endDate to date "{end_date}"\n'
        '  repeat with c in calendars\n'
        '    set evts to (every event of c whose start date ≥ startDate '
        'and start date ≤ endDate)\n'
        '    repeat with e in evts\n'
        '      set evtName to summary of e\n'
        '      set evtStart to start date of e\n'
        '      set evtEnd to end date of e\n'
        '      set output to output & evtName & " | " '
        '& (evtStart as string) & " — " & (evtEnd as string) & linefeed\n'
        '    end repeat\n'
        '  end repeat\n'
        'end tell\n'
        'return output'
    )

    ok, msg = _run_applescript(script)
    if ok:
        if not msg.strip():
            return f"📅 No events found for {display_date}."
        header = f"📅 Events for {display_date}:\n"
        lines = [f"  • {line}" for line in msg.strip().splitlines() if line.strip()]
        return header + "\n".join(lines)
    return f"❌ Failed to get calendar events: {msg}"


def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str = "",
    location: str = "",
    notes: str = "",
    calendar_name: str = "",
) -> str:
    """Create a new event in macOS Calendar.

    Parameters
    ----------
    title : str          Event summary.
    start_time : str     Start time (same formats as set_alarm).
    end_time : str       End time.  Defaults to start + 1 hour.
    location : str       Optional location string.
    notes : str          Optional event notes.
    calendar_name : str  Which calendar to use (empty = first writable).
    """
    if not _is_macos():
        return "⚠️  Creating calendar events is only supported on macOS."

    start_dt = _parse_time(start_time)
    if start_dt is None:
        return f"❌ Could not parse start time '{start_time}'."

    if end_time:
        end_dt = _parse_time(end_time)
        if end_dt is None:
            end_dt = start_dt + timedelta(hours=1)
    else:
        end_dt = start_dt + timedelta(hours=1)

    as_start = start_dt.strftime("%B %d, %Y at %H:%M:%S")
    as_end = end_dt.strftime("%B %d, %Y at %H:%M:%S")
    safe_title = _escape(title)
    safe_location = _escape(location)
    safe_notes = _escape(notes)

    # Build properties
    props = (
        f'summary:"{safe_title}", '
        f'start date:date "{as_start}", '
        f'end date:date "{as_end}"'
    )
    if location:
        props += f', location:"{safe_location}"'
    if notes:
        props += f', description:"{safe_notes}"'

    # Target calendar
    if calendar_name:
        cal_target = f'calendar "{_escape(calendar_name)}"'
    else:
        # Use the first writable calendar
        cal_target = "first calendar"

    script = (
        'tell application "Calendar"\n'
        f'  tell {cal_target}\n'
        f'    make new event at end with properties {{{props}}}\n'
        '  end tell\n'
        'end tell'
    )

    ok, msg = _run_applescript(script)
    if ok:
        friendly_start = start_dt.strftime("%Y-%m-%d %I:%M %p")
        friendly_end = end_dt.strftime("%I:%M %p")
        return f'✅ Event created: "{title}" on {friendly_start} – {friendly_end}'
    return f"❌ Failed to create event: {msg}"


# ═══════════════════════════════════════════════════════════════════
#  3.  Notes  (fixed double-text bug)
# ═══════════════════════════════════════════════════════════════════
#
# BUG FIX: The old code set both `name` AND `body` properties, but
# Notes.app auto-generates the body's first line from `name` — so the
# title appeared twice.  Fix: set only `body` with an HTML <h1> title
# line, letting Notes derive `name` from it automatically.

def write_note(content: str, title: str = "") -> str:
    """Create a new note in macOS Notes app (no duplicate text)."""
    if not _is_macos():
        return "⚠️  Writing notes is only supported on macOS."
    if not content.strip():
        return "❌ Cannot create an empty note."

    if not title:
        title = content.strip().splitlines()[0][:80]

    # Build an HTML body: Notes.app uses the first block as the title.
    # Providing ONE body string avoids the name+body duplication.
    safe_title = _escape(title)
    safe_content = _escape(content)
    html_body = f"<h1>{safe_title}</h1><br>{safe_content}"

    # Try iCloud account first, fall back to default
    for target in (
        'tell account "iCloud"\n  make new note at folder "Notes" with properties {{body:"{body}"}}\nend tell',
        'make new note with properties {{body:"{body}"}}',
    ):
        script = (
            'tell application "Notes"\n'
            f'  {target.format(body=html_body)}\n'
            'end tell'
        )
        ok, msg = _run_applescript(script)
        if ok:
            return f'📝 Note created: "{title}"'

    return f"❌ Failed to create note: {msg}"


# ═══════════════════════════════════════════════════════════════════
#  4.  Timer / Countdown  (pure Python — Clock.app has no AS dict)
# ═══════════════════════════════════════════════════════════════════

def run_timer(duration_str: str) -> str:
    """Run a blocking countdown timer.  Prints progress to stdout.

    For async callers wrap with ``asyncio.to_thread(run_timer, ...)``.
    """
    total = _parse_seconds(duration_str)
    if total is None:
        return f"❌ Could not parse duration '{duration_str}'. Try '5 min', '90 seconds', '1 hour'."
    if total <= 0:
        return "❌ Duration must be positive."
    if total > 86400:
        return "⚠️  Maximum timer duration is 24 hours. For longer durations use a Reminder instead."

    print(f"⏱️  Timer started: {_fmt_duration(total)}")
    remaining = total
    try:
        while remaining > 0:
            print(f"\r  ⏳ {_fmt_duration(remaining)} remaining  ", end="", flush=True)
            _time.sleep(min(remaining, 1))
            remaining -= 1
        print(f"\r  ✅ Timer done! ({_fmt_duration(total)} elapsed)   ")
    except KeyboardInterrupt:
        elapsed = total - remaining
        print(f"\n  ⚠️  Timer cancelled after {_fmt_duration(elapsed)}.")
        return f"⏱️  Timer cancelled after {_fmt_duration(elapsed)}."

    # macOS notification
    if _is_macos():
        _run_applescript(
            'display notification "Timer complete!" '
            f'with title "⏱️ {_fmt_duration(total)} Timer" sound name "Glass"'
        )

    return f"⏱️  Timer complete: {_fmt_duration(total)}"


def _fmt_duration(secs: int) -> str:
    """Format seconds into a human-friendly string."""
    if secs >= 3600:
        h, r = divmod(secs, 3600)
        m, s = divmod(r, 60)
        return f"{h}h {m:02d}m {s:02d}s"
    if secs >= 60:
        m, s = divmod(secs, 60)
        return f"{m}m {s:02d}s"
    return f"{secs}s"


# ═══════════════════════════════════════════════════════════════════
#  5.  Stopwatch  (pure Python)
# ═══════════════════════════════════════════════════════════════════

def run_stopwatch() -> str:
    """Interactive stopwatch.  Press Enter for lap, Ctrl+C to stop."""
    print("⏱️  Stopwatch started.  Press Enter for a lap, Ctrl+C to stop.")
    start = _time.time()
    laps: list[float] = []

    try:
        while True:
            input()  # blocks until Enter
            elapsed = _time.time() - start
            laps.append(elapsed)
            print(f"  🏁 Lap {len(laps)}: {_fmt_duration(int(elapsed))}")
    except (KeyboardInterrupt, EOFError):
        total = _time.time() - start
        print(f"\n  ⏹️  Stopwatch stopped: {_fmt_duration(int(total))}")
        if laps:
            lap_lines = "\n".join(
                f"  Lap {i+1}: {_fmt_duration(int(t))}" for i, t in enumerate(laps)
            )
            return f"⏱️  Total: {_fmt_duration(int(total))}\n{lap_lines}"
        return f"⏱️  Total: {_fmt_duration(int(total))}"


# ═══════════════════════════════════════════════════════════════════
#  6.  World Clock  (pure Python + zoneinfo)
# ═══════════════════════════════════════════════════════════════════
#
# Uses stdlib zoneinfo (Python 3.9+).  Falls back to pytz if available.

# Common city → IANA timezone mapping for natural queries
_CITY_TZ: dict[str, str] = {
    "new york": "America/New_York", "nyc": "America/New_York",
    "los angeles": "America/Los_Angeles", "la": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles", "sf": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "tokyo": "Asia/Tokyo",
    "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai",
    "singapore": "Asia/Singapore",
    "sydney": "Australia/Sydney",
    "dubai": "Asia/Dubai",
    "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata", "india": "Asia/Kolkata",
    "seoul": "Asia/Seoul",
    "moscow": "Europe/Moscow",
    "são paulo": "America/Sao_Paulo", "sao paulo": "America/Sao_Paulo",
    "ulaanbaatar": "Asia/Ulaanbaatar", "ulan bator": "Asia/Ulaanbaatar",
}


def world_clock(city_or_tz: str) -> str:
    """Show current time in a city / timezone.

    Accepts city names ("Tokyo", "New York") or IANA tz ("Asia/Tokyo").
    """
    tz_name = _CITY_TZ.get(city_or_tz.strip().lower())
    if not tz_name:
        # Maybe it's already an IANA string
        tz_name = city_or_tz.strip()

    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except (ImportError, KeyError):
        try:
            import pytz
            tz = pytz.timezone(tz_name)  # type: ignore[assignment]
        except Exception:
            return (
                f"❌ Unknown timezone or city '{city_or_tz}'.  "
                "Try a city name (Tokyo, London) or IANA zone (Asia/Tokyo)."
            )

    now = datetime.now(tz)
    return (
        f"🌍 {city_or_tz.title()}: {now.strftime('%A, %B %d, %Y %I:%M %p')} "
        f"({tz_name})"
    )


# ═══════════════════════════════════════════════════════════════════
#  Task agent graph node
# ═══════════════════════════════════════════════════════════════════

def task_agent_node(state: dict) -> dict:
    """Multi-agent graph node — detects and executes macOS / utility tasks.

    The coordinator places ``task_agent`` in the plan.  This node inspects
    the query, dispatches to the right function, and appends to agent_results.
    """
    from src.agents import AgentResult

    query: str = state.get("translated_query") or state["query"]
    q = query.lower()

    result_text = ""
    sources: list[str] = []

    # ── Dispatch ────────────────────────────────────────────────
    if _match_alarm(q):
        time_str, title = _extract_alarm_params(query)
        result_text = set_alarm(time_str, title)
        sources = ["macOS Reminders"]

    elif _match_create_event(q):
        title, start, end, loc, notes = _extract_event_params(query)
        result_text = create_calendar_event(title, start, end, loc, notes)
        sources = ["macOS Calendar"]

    elif _match_calendar(q):
        date_str = _extract_date_param(query)
        result_text = get_calendar_events(date_str)
        sources = ["macOS Calendar"]

    elif _match_note(q):
        content, title = _extract_note_params(query)
        result_text = write_note(content, title)
        sources = ["macOS Notes"]

    elif _match_timer(q):
        dur = _extract_duration(query)
        result_text = run_timer(dur)
        sources = ["Python Timer"]

    elif _match_stopwatch(q):
        result_text = run_stopwatch()
        sources = ["Python Stopwatch"]

    elif _match_world_clock(q):
        city = _extract_city(query)
        result_text = world_clock(city)
        sources = ["World Clock"]

    else:
        result_text = (
            "I detected a task intent but couldn't determine the specific action.\n"
            "Supported: reminder/alarm, calendar events (get/create), notes, "
            "timer, stopwatch, world clock."
        )

    agent_result: AgentResult = {
        "agent": "task_agent",
        "content": result_text,
        "sources": sources,
        "confidence": "high" if any(result_text.startswith(p) for p in ("✅", "📅", "📝", "⏱️", "🌍")) else "low",
    }

    prior = state.get("agent_results", [])
    used = state.get("agents_used", [])

    return {
        "agent_results": prior + [agent_result],
        "agents_used": used + ["🖥️  Task Agent"],
    }


# ═══════════════════════════════════════════════════════════════════
#  Intent matchers  (public: matches_any_task)
# ═══════════════════════════════════════════════════════════════════

def _match_alarm(text: str) -> bool:
    return bool(re.search(
        r"(set\s+(an?\s+)?alarm|set\s+(an?\s+)?reminder|remind\s+me|wake\s+me)",
        text, re.I,
    ))

def _match_create_event(text: str) -> bool:
    return bool(re.search(
        r"(create|add|schedule|set up|book|make)\s+(an?\s+)?(event|meeting|appointment|call)",
        text, re.I,
    ))

def _match_calendar(text: str) -> bool:
    return bool(re.search(
        r"(calendar|schedule|events?\s+(for|on|today|tomorrow)|my\s+events"
        r"|what('s| is)\s+on\s+my|show\s+(my\s+)?calendar)",
        text, re.I,
    ))

def _match_note(text: str) -> bool:
    return bool(re.search(
        r"(write\s+(a\s+)?note|create\s+(a\s+)?note|save\s+(a\s+)?note"
        r"|take\s+(a\s+)?note|note\s+down|jot\s+down)",
        text, re.I,
    ))

def _match_timer(text: str) -> bool:
    return bool(re.search(
        r"(set\s+(a\s+)?timer|start\s+(a\s+)?timer|countdown|timer\s+for)",
        text, re.I,
    ))

def _match_stopwatch(text: str) -> bool:
    return bool(re.search(r"(start\s+(a\s+)?stopwatch|stopwatch)", text, re.I))

def _match_world_clock(text: str) -> bool:
    return bool(re.search(
        r"(what\s+time\s+(is\s+it\s+)?in\s+|time\s+in\s+|world\s+clock|current\s+time\s+in\s+)",
        text, re.I,
    ))


def matches_any_task(text: str) -> bool:
    """Return True if the text matches any known task intent.

    Used by the coordinator to detect action queries.
    """
    low = text.lower()
    return (
        _match_alarm(low) or _match_create_event(low) or _match_calendar(low)
        or _match_note(low) or _match_timer(low) or _match_stopwatch(low)
        or _match_world_clock(low)
    )


# ═══════════════════════════════════════════════════════════════════
#  Parameter extraction helpers
# ═══════════════════════════════════════════════════════════════════

def _extract_alarm_params(query: str) -> tuple[str, str]:
    """Extract time and title from a reminder/alarm query."""
    time_str = ""
    title = "AI Assistant Reminder"

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
            m = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2})", query)
            if m:
                time_str = m.group(1)
            else:
                m = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)", query)
                if m:
                    time_str = m.group(1)
                else:
                    time_str = "5 minutes"

    # "to <title>" / "called <title>" / "named <title>"
    m = re.search(
        r"(?:to|called|named|titled|about|for)\s+['\"]?(.+?)['\"]?\s*$",
        query, re.I,
    )
    if m:
        candidate = m.group(1).strip().rstrip(".")
        if len(candidate) > 2 and not re.search(r"\d+\s*min|\d+\s*hour|\d:\d|am|pm", candidate, re.I):
            title = candidate

    return time_str, title


def _extract_event_params(query: str) -> tuple[str, str, str, str, str]:
    """Extract (title, start, end, location, notes) from a create-event query."""
    title = "New Event"
    start = ""
    end = ""
    location = ""
    notes = ""

    # title: "called X" / "titled X" / quoted text
    m = re.search(r'(?:called|titled|named)\s+["\']?(.+?)["\']?\s*(?:at|on|from|$)', query, re.I)
    if m:
        title = m.group(1).strip()
    else:
        # Quoted title
        m = re.search(r'["\'](.+?)["\']', query)
        if m:
            title = m.group(1)

    # start: "at HH:MM" / "on YYYY-MM-DD" / "tomorrow at ..."
    m = re.search(r"(?:at|from)\s+(\d{1,2}:\d{2}\s*(?:AM|PM)?)", query, re.I)
    if m:
        start = m.group(1)
    # date prefix: "tomorrow", "on 2026-03-01"
    m_date = re.search(r"(?:on|for)\s+(tomorrow|\d{4}-\d{2}-\d{2})", query, re.I)
    if m_date and start:
        day = m_date.group(1)
        if day.lower() == "tomorrow":
            tmr = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            start = f"{tmr} {start}"
        else:
            start = f"{day} {start}"
    if not start:
        start = "1 hour"  # default: event starts in 1 hour

    # end: "to HH:MM" / "until HH:MM"
    m = re.search(r"(?:to|until|till|-)\s*(\d{1,2}:\d{2}\s*(?:AM|PM)?)", query, re.I)
    if m:
        end = m.group(1)

    # location: "at <place>" (after time extraction to avoid conflicts)
    m = re.search(r"(?:location|place|at|in)\s+([A-Z][a-zA-Z\s,]+?)(?:\s+(?:from|at|on|$))", query)
    if m:
        location = m.group(1).strip()

    return title, start, end, location, notes


def _extract_date_param(query: str) -> str:
    """Extract a date from a calendar retrieval query."""
    low = query.lower()
    if "tomorrow" in low:
        return "tomorrow"
    if "yesterday" in low:
        return "yesterday"
    if "next week" in low:
        return "next week"
    m = re.search(r"(\d{4}-\d{2}-\d{2})", query)
    if m:
        return m.group(1)
    m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", query)
    if m:
        return m.group(1)
    return "today"


def _extract_note_params(query: str) -> tuple[str, str]:
    """Extract note content and optional title."""
    title = ""
    content = query

    # Strip the command prefix: "write a note: <content>"
    m = re.search(
        r"(?:note|write\s+a\s+note|save\s+a\s+note|take\s+a\s+note|jot\s+down)[:\s]+(.+)",
        query, re.I | re.S,
    )
    if m:
        content = m.group(1).strip()

    # "titled <title>: <content>"
    m = re.search(r"titled?\s+['\"]?(.+?)['\"]?\s*:\s*(.+)", content, re.I | re.S)
    if m:
        title = m.group(1).strip()
        content = m.group(2).strip()

    return content, title


def _extract_duration(query: str) -> str:
    """Extract timer duration from query."""
    m = re.search(r"(?:for|of)?\s*(\d+\s*(?:min(?:ute)?s?|hours?|hrs?|secs?|seconds?))", query, re.I)
    if m:
        return m.group(1)
    m = re.search(r"(\d+)\s*(?:min|sec|hour|hr)", query, re.I)
    if m:
        return m.group(0)
    return "5 minutes"


def _extract_city(query: str) -> str:
    """Extract city/timezone from a world-clock query."""
    m = re.search(r"(?:time\s+in|what\s+time\s+(?:is\s+it\s+)?in)\s+(.+?)(?:\?|$)", query, re.I)
    if m:
        return m.group(1).strip().rstrip("?.")
    m = re.search(r"world\s+clock\s+(.+?)(?:\?|$)", query, re.I)
    if m:
        return m.group(1).strip().rstrip("?.")
    return "UTC"
