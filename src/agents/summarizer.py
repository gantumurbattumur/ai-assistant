"""Summarizer agent — merges outputs from multiple agents into one response.

Coordinator mode: merges multi-agent results.
Standalone mode: summarize_text(text) → simple direct SDK call.
"""
from __future__ import annotations

from src.agents import MultiAgentState, AgentResult
from src.config import MODEL_NAME, get_openai_client


# ── Standalone (simple direct call) ─────────────────────────────

def summarize_text(text: str) -> str:
    """Summarize text or fetched URL content. Simple direct SDK call."""
    resp = get_openai_client().chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a summarization expert. Provide a clear, concise summary "
                    "with key points as bullet points. Use markdown formatting."
                ),
            },
            {"role": "user", "content": f"Summarize the following:\n\n{text}"},
        ],
    )
    return resp.choices[0].message.content or ""


# ── Coordinator pipeline node ───────────────────────────────────


def summarizer_node(state: MultiAgentState) -> dict:
    """Merge all agent outputs into a coherent final answer."""
    query = state.get("translated_query") or state["query"]
    prior = state.get("agent_results", [])
    agents_used = state.get("agents_used", [])

    # Collect content from all content-producing agents
    sections = []
    all_sources = []
    for r in prior:
        if r["content"] and r["agent"] != "translator":
            label = r["agent"].capitalize()
            sections.append(f"## From {label}:\n{r['content']}")
            all_sources.extend(r.get("sources", []))

    if not sections:
        return {
            "response": "No agents produced content to summarize.",
            "agents_used": agents_used + ["📝 Summarizer (skipped — no content)"],
        }

    if len(sections) == 1:
        # Only one source → just pass through, no need to merge.
        # Find the actual content-producing result (not translator).
        content_result = next(
            (r for r in prior if r["content"] and r["agent"] != "translator"),
            None,
        )
        return {
            "response": content_result["content"] if content_result else "",
            "agents_used": agents_used + ["📝 Summarizer (passthrough)"],
        }

    combined = "\n\n".join(sections)

    resp = get_openai_client().chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a summarizer. You receive findings from multiple research agents. "
                    "Merge them into ONE coherent, well-structured answer. Use markdown formatting. "
                    "If sources agree, present a unified answer. "
                    "If sources disagree, highlight the differences clearly. "
                    "Include citations/sources where available."
                ),
            },
            {
                "role": "user",
                "content": f"User question: {query}\n\n{combined}",
            },
        ],
    )
    content = resp.choices[0].message.content or ""

    # Append source list if available
    if all_sources:
        source_list = "\n".join(f"- {s}" for s in dict.fromkeys(all_sources))
        content += f"\n\n---\n**Sources:**\n{source_list}"

    result: AgentResult = {
        "agent": "summarizer",
        "content": content,
        "sources": all_sources,
        "confidence": "high",
    }

    return {
        "response": content,
        "agent_results": prior + [result],
        "agents_used": agents_used + ["📝 Summarizer"],
    }
