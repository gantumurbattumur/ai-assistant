"""Translator agent — detects language and translates query/response.

Two modes in coordinator pipeline:
  - First in plan → translate query to English (input mode)
  - Last in plan  → translate final response back to user's language (output mode)

Standalone mode:
  translate_text(text, target_language) → simple direct SDK call
"""
from __future__ import annotations

from openai import OpenAI

from src.agents import MultiAgentState, AgentResult


# ── Standalone (simple direct call) ─────────────────────────────

def translate_text(text: str, target_language: str = "English") -> str:
    """Translate text to target language. Simple direct SDK call."""
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a professional translator. Translate the given text to {target_language}. "
                    "Only output the translation, nothing else."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content or ""


# ── Coordinator pipeline node ───────────────────────────────────


def translator_node(state: MultiAgentState) -> dict:
    """Translate query or response depending on position in plan."""
    query = state["query"]
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    prior = state.get("agent_results", [])
    agents_used = state.get("agents_used", [])

    client = OpenAI()

    # Determine mode: input (first) or output (last)
    translator_positions = [i for i, a in enumerate(plan) if a == "translator"]
    is_input_mode = current_step == translator_positions[0] if translator_positions else True

    if is_input_mode:
        # ── Input mode: translate query to English ──────────────
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Detect the language of the input text, then translate it to English. "
                        "Return ONLY the English translation. If it's already English, return it as-is."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        translated = resp.choices[0].message.content or query

        result: AgentResult = {
            "agent": "translator",
            "content": f"Translated to English: {translated}",
            "sources": [],
            "confidence": "high",
        }

        return {
            "translated_query": translated,
            "agent_results": prior + [result],
            "agents_used": agents_used + ["🌍 Translator (→ English)"],
        }

    else:
        # ── Output mode: translate response back to user's language ─
        # Find the current response from the most recent content agent
        response_so_far = state.get("response", "")
        if not response_so_far:
            # Grab from the last agent's output
            content_results = [r for r in prior if r["content"] and r["agent"] != "translator"]
            if content_results:
                response_so_far = content_results[-1]["content"]

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the following text to the same language as the user's "
                        "original query below. Preserve all formatting (markdown, bullet "
                        "points, etc). Return ONLY the translation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original query (detect target language from this): {query}\n\n"
                        f"Text to translate:\n{response_so_far}"
                    ),
                },
            ],
        )
        translated = resp.choices[0].message.content or response_so_far

        result = {
            "agent": "translator",
            "content": translated,
            "sources": [],
            "confidence": "high",
        }

        return {
            "response": translated,
            "agent_results": prior + [result],
            "agents_used": agents_used + ["🌍 Translator (→ user language)"],
        }
