"""Coordinator agent — plans which agents handle the query.

Strategy (Option C — Hybrid):
  1. Rule-based pattern matching for common intents.
  2. Falls back to an LLM call if no rule matches clearly.
"""
from __future__ import annotations

import re
from typing import List

from src.agents import MultiAgentState
from src.config import MODEL_NAME, get_openai_client


# ── Rule-based patterns ────────────────────────────────────────────
_PATTERNS: list[tuple[re.Pattern, list[str], str]] = [
    # Book + verify (MUST come before book-only to match first)
    (
        re.compile(r"(verify|fact.?check|still true|still accurate|still relevant|is it correct)", re.I),
        ["librarian", "researcher", "critic", "summarizer"],
        "User wants to verify a book claim → full pipeline",
    ),
    # Book-only queries
    (
        re.compile(r"(my book|my document|my pdf|my epub|in the book|from the book)", re.I),
        ["librarian"],
        "Query mentions personal books → Librarian only",
    ),
    # Translation
    (
        re.compile(r"(translate|翻訳|перевод|орчуул)", re.I),
        ["translator"],
        "Explicit translation request",
    ),
    # Summarize
    (
        re.compile(r"(summarize|summary|тоймло|give me the gist)", re.I),
        ["summarizer"],
        "Explicit summarization request",
    ),
    # Web search
    (
        re.compile(r"(search|latest|current|today|news|trending|2024|2025|2026)", re.I),
        ["researcher"],
        "User wants current/web information → Researcher",
    ),
]

# ── Task-detection helper (imported from tasks module) ──────────
def _is_task_query(query: str) -> bool:
    """Check if the query is a macOS task (alarm, calendar, note)."""
    try:
        from src.tasks.macos_agent import matches_any_task
        return matches_any_task(query)
    except ImportError:
        return False

# Non-English detection (simple heuristic: if >40% non-ASCII → non-English)
_NON_ASCII_THRESHOLD = 0.30


def _detect_language(text: str) -> str:
    """Cheap language detection. Returns 'en' or 'other'."""
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / max(len(text), 1)
    return "other" if ratio > _NON_ASCII_THRESHOLD else "en"


def _rule_based_plan(query: str) -> tuple[list[str], str] | None:
    """Try to match a rule. Returns (plan, reasoning) or None."""
    # Check task intents first (alarm, calendar, note)
    if _is_task_query(query):
        # Hybrid: if summarize + calendar → task_agent then summarizer
        q_low = query.lower()
        if any(kw in q_low for kw in ("summarize", "summary", "give me the gist")):
            return ["task_agent", "summarizer"], "Task action + summarization"
        return ["task_agent"], "macOS task detected"

    for pattern, plan, reasoning in _PATTERNS:
        if pattern.search(query):
            return plan, reasoning
    return None


def _llm_plan(query: str, language: str) -> tuple[list[str], str]:
    """Fall back to LLM to decide the plan."""
    system = (
        "You are a task planner for a multi-agent AI system. "
        "Available agents:\n"
        "  - librarian: searches the user's personal book/document collection\n"
        "  - researcher: searches the live web for current information\n"
        "  - translator: translates text between languages\n"
        "  - summarizer: condenses and merges information from multiple sources\n"
        "  - critic: reviews answers for quality, contradictions, and accuracy\n"
        "  - task_agent: performs macOS tasks (set alarms/reminders, get calendar events, write notes)\n\n"
        "Given the user's query, return ONLY a JSON object with two keys:\n"
        '  "plan": list of agent names in execution order\n'
        '  "reasoning": one-sentence explanation\n\n'
        "Rules:\n"
        "- Use the fewest agents necessary.\n"
        "- If the query is simple (greeting, basic fact), use an empty plan [].\n"
        "- If the query involves the user's books, always include librarian.\n"
        "- If the query needs web data, include researcher.\n"
        "- If multiple agents produce output, end with summarizer.\n"
        "- If the user wants to verify something, include critic before summarizer.\n"
        f"- Detected query language: {language}\n"
        "- If query is non-English and the task also requires other agents, "
        "  start with translator and end with translator.\n"
    )

    response = get_openai_client().chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
    )

    import json
    raw = response.choices[0].message.content or "{}"
    result = json.loads(raw)
    plan = result.get("plan", [])
    reasoning = result.get("reasoning", "LLM-planned")

    # Validate agent names
    valid = {"librarian", "researcher", "translator", "summarizer", "critic", "task_agent"}
    plan = [a for a in plan if a in valid]

    return plan, reasoning


# ── Public node function ───────────────────────────────────────────

def coordinator_node(state: MultiAgentState) -> dict:
    """Coordinator: analyse the query and produce an execution plan."""
    query = state["query"]
    language = _detect_language(query)

    # Try rules first (fast, free)
    result = _rule_based_plan(query)

    if result:
        plan, reasoning = result
    else:
        # LLM fallback (slower, costs tokens)
        plan, reasoning = _llm_plan(query, language)

    # If non-English and plan involves non-translator agents,
    # bookend with translator
    if language != "en" and plan and plan[0] != "translator":
        plan = ["translator"] + plan
        if plan[-1] != "translator":
            plan.append("translator")
        reasoning += " (auto-added translator for non-English query)"

    # If 2+ agents produce content and summarizer not included, add it
    content_agents = [a for a in plan if a in ("librarian", "researcher")]
    if len(content_agents) >= 2 and "summarizer" not in plan:
        # insert summarizer at end (before final translator if present)
        if plan and plan[-1] == "translator":
            plan.insert(-1, "summarizer")
        else:
            plan.append("summarizer")

    return {
        "plan": plan,
        "plan_reasoning": reasoning,
        "language": language,
        "current_step": 0,
        "agent_results": [],
        "agents_used": [],
        "needs_human_confirm": False,
        "should_stop": False,
    }
