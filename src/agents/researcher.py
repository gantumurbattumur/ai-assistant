"""Researcher agent — searches the live web for current information.

Coordinator mode: targeted search using prior agent context.
Standalone mode: search_web(query) → simple direct Tavily + SDK call.
"""
from __future__ import annotations

from src.agents import MultiAgentState, AgentResult
from src.core import get_web_search_tool
from src.config import MODEL_NAME, get_openai_client


# ── Standalone (simple direct call) ─────────────────────────────

def search_web(query: str, num_results: int = 3) -> list[dict]:
    """Search the web and return raw results. Simple direct call."""
    tool = get_web_search_tool()
    results = tool.invoke({"query": query})
    return results[:num_results] if results else []


def search_and_summarize(query: str) -> str:
    """Search the web and synthesize an answer. Direct SDK call."""
    results = search_web(query, num_results=3)
    if not results:
        return "No results found on the web."

    web_context = "\n\n---\n\n".join(
        f"URL: {r.get('url', 'N/A')}\n{r.get('content', '')}" for r in results
    )

    resp = get_openai_client().chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a researcher. Summarise the web search results to answer "
                    "the user's question. Cite URLs where relevant. Be factual and concise."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nWeb results:\n{web_context}",
            },
        ],
    )
    return resp.choices[0].message.content or ""


# ── Coordinator pipeline node ───────────────────────────────────


def _build_search_query(query: str, prior_results: list[AgentResult]) -> str:
    """If Librarian already found something, make the web search targeted."""
    librarian_result = next(
        (r for r in prior_results if r["agent"] == "librarian" and r["content"]),
        None,
    )
    if librarian_result:
        resp = get_openai_client().chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Given a user question and what was found in their books, "
                        "write a concise web search query to find additional evidence, "
                        "current data, or verification. Return ONLY the search query."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Book finding: {librarian_result['content'][:500]}"
                    ),
                },
            ],
        )
        return resp.choices[0].message.content or query
    return query


def researcher_node(state: MultiAgentState) -> dict:
    """Search the web, synthesise findings."""
    query = state.get("translated_query") or state["query"]
    prior = state.get("agent_results", [])

    # ── Build targeted search query ─────────────────────────────
    search_query = _build_search_query(query, prior)

    # ── Web search ──────────────────────────────────────────────
    tool = get_web_search_tool()
    raw_results = tool.invoke({"query": search_query})

    if not raw_results:
        result: AgentResult = {
            "agent": "researcher",
            "content": "",
            "sources": [],
            "confidence": "none",
        }
        agents_used = state.get("agents_used", []) + ["🔍 Researcher"]
        return {
            "agent_results": prior + [result],
            "agents_used": agents_used,
            "needs_human_confirm": True,
            "human_confirm_message": (
                "🔍 Researcher found nothing on the web. "
                "Should I try rephrasing and searching again?"
            ),
        }

    # ── Synthesise with direct SDK ──────────────────────────────
    web_context = "\n\n---\n\n".join(
        f"URL: {r.get('url', 'N/A')}\n{r.get('content', '')}" for r in raw_results
    )
    urls = [r.get("url", "") for r in raw_results if r.get("url")]

    resp = get_openai_client().chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a researcher. Summarise the web search results to answer "
                    "the user's question. Cite URLs where relevant. "
                    "Be factual and concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Web results:\n{web_context}"
                ),
            },
        ],
    )
    content = resp.choices[0].message.content or ""

    result = {
        "agent": "researcher",
        "content": content,
        "sources": urls,
        "confidence": "high" if len(raw_results) >= 2 else "medium",
    }

    agents_used = state.get("agents_used", []) + ["🔍 Researcher"]

    return {
        "agent_results": prior + [result],
        "agents_used": agents_used,
    }
