"""Librarian agent — searches the user's personal book collection via RAG.

Coordinator mode: returns AgentResult with context for other agents.
Standalone mode: search_books(question) → simple direct answer string.
"""
from __future__ import annotations

from src.agents import MultiAgentState, AgentResult
from src.core import create_vectorstore, create_retrieval_grader
from src.config import MODEL_NAME, get_openai_client

_SYSTEM_PROMPT = (
    "You are a librarian. Using ONLY the provided book excerpts, "
    "answer the user's question. Cite which passage you used. "
    "If the excerpts don't contain the answer, say so honestly."
)


# ── Shared helper ────────────────────────────────────────────────

def _retrieve_and_grade(question: str) -> list:
    """Retrieve documents from the vectorstore and filter to relevant ones."""
    retriever = create_vectorstore()
    documents = retriever.invoke(question)
    grader = create_retrieval_grader()
    relevant = []
    for doc in documents:
        score = grader.invoke({"question": question, "document": doc.page_content})
        if score.binary_score == "yes":  # type: ignore[union-attr]
            relevant.append(doc)
    return relevant


def _answer_from_docs(question: str, relevant_docs: list) -> str:
    """Synthesise an answer from relevant docs using the LLM."""
    context = "\n\n---\n\n".join(d.page_content for d in relevant_docs)
    resp = get_openai_client().chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Book excerpts:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content or "No answer could be generated."


# ── Standalone (simple direct call) ─────────────────────────────

def search_books(question: str) -> str:
    """Search books and return a direct answer. Simple call, no graph state."""
    relevant_docs = _retrieve_and_grade(question)
    if not relevant_docs:
        return "No relevant information found in your books."
    return _answer_from_docs(question, relevant_docs)


# ── Coordinator pipeline node ───────────────────────────────────


def librarian_node(state: MultiAgentState) -> dict:
    """Search books, grade relevance, return findings."""
    query = state.get("translated_query") or state["query"]
    prior = state.get("agent_results", [])

    relevant_docs = _retrieve_and_grade(query)

    if relevant_docs:
        content = _answer_from_docs(query, relevant_docs)
        sources = list({d.metadata.get("source", "unknown") for d in relevant_docs})
        confidence = "high" if len(relevant_docs) >= 2 else "medium"
    else:
        content = ""
        sources = []
        confidence = "none"

    result: AgentResult = {
        "agent": "librarian",
        "content": content,
        "sources": sources,
        "confidence": confidence,
    }

    agents_used = state.get("agents_used", []) + ["📚 Librarian"]

    return {
        "agent_results": prior + [result],
        "agents_used": agents_used,
        # If nothing found → flag for human confirmation
        "needs_human_confirm": confidence == "none",
        "human_confirm_message": (
            "📚 Librarian found nothing in your books. "
            "Should I search the web instead?"
            if confidence == "none"
            else ""
        ),
    }
