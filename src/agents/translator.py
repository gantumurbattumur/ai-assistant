"""Translator agent — detects language and translates query/response.

Two modes in coordinator pipeline:
  - First in plan → translate query to English (input mode)
  - Last in plan  → translate final response back to user's language (output mode)

Standalone mode:
  translate_text(text, target_language) → simple direct SDK call
"""
from __future__ import annotations

from src.agents import MultiAgentState, AgentResult
from src.config import MODEL_NAME, get_openai_client


# ── Standalone (simple direct call) ─────────────────────────────

def translate_text(text: str, target_language: str = "English") -> str:
    """Translate text to target language. Simple direct SDK call."""
    resp = get_openai_client().chat.completions.create(
        model=MODEL_NAME,
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
    # current_step is the index of THIS invocation in the plan (not yet incremented
    # by dispatcher — dispatcher increments AFTER the agent returns).
    current_step = state.get("current_step", 0)
    prior = state.get("agent_results", [])
    agents_used = state.get("agents_used", [])

    # Determine mode: "input" if this is the FIRST translator in the plan,
    # "output" if it's the LAST.  Use plan-index comparison, not current_step
    # arithmetic, so the logic is explicit and immune to dispatcher ordering.
    translator_indices = [i for i, a in enumerate(plan) if a == "translator"]
    first_translator_idx = translator_indices[0] if translator_indices else 0
    is_input_mode = current_step == first_translator_idx

    if is_input_mode:
        # ── Input mode: translate query to English ──────────────
        resp = get_openai_client().chat.completions.create(
            model=MODEL_NAME,
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

        resp = get_openai_client().chat.completions.create(
            model=MODEL_NAME,
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
