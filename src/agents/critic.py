"""Critic agent — reviews answers for quality AND compares sources (Option C).

Two jobs:
  1. Quality check: Is the answer complete, well-structured, free of hallucination?
  2. Source comparison: Do the Librarian and Researcher agree or contradict?
"""
from __future__ import annotations

from openai import OpenAI

from src.agents import MultiAgentState, AgentResult


def critic_node(state: MultiAgentState) -> dict:
    """Review agent outputs for quality and cross-source consistency."""
    query = state.get("translated_query") or state["query"]
    prior = state.get("agent_results", [])
    agents_used = state.get("agents_used", [])

    # Gather what each agent said
    agent_outputs = {}
    for r in prior:
        if r["content"] and r["agent"] != "translator":
            agent_outputs[r["agent"]] = r["content"]

    if not agent_outputs:
        return {
            "agents_used": agents_used + ["⚖️ Critic (skipped — no content to review)"],
        }

    # Build the review prompt
    sections = "\n\n---\n\n".join(
        f"**{name.capitalize()}** said:\n{text}"
        for name, text in agent_outputs.items()
    )

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a critical reviewer for a multi-agent AI system. "
                    "You have TWO jobs:\n\n"
                    "1. **Quality check**: Is each answer complete? Is there evidence of "
                    "hallucination or unsupported claims? Is it well-structured?\n\n"
                    "2. **Source comparison**: Do the different agents' answers agree or "
                    "contradict each other? Highlight any discrepancies.\n\n"
                    "Output a brief review in markdown with sections:\n"
                    "## Quality Assessment\n"
                    "## Source Comparison\n"
                    "## Recommendation\n"
                    "(say 'consistent' if sources agree, 'contradicts' if they don't, "
                    "and explain what to trust)"
                ),
            },
            {
                "role": "user",
                "content": f"User question: {query}\n\nAgent outputs:\n{sections}",
            },
        ],
    )
    review = resp.choices[0].message.content or ""

    result: AgentResult = {
        "agent": "critic",
        "content": review,
        "sources": [],
        "confidence": "high",
    }

    return {
        "agent_results": prior + [result],
        "agents_used": agents_used + ["⚖️ Critic"],
    }
