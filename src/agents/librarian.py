"""Librarian agent — searches the user's personal book collection via RAG.

Coordinator mode: returns AgentResult with context for other agents.
Standalone mode: search_books(question) → simple direct answer string.
"""
from __future__ import annotations

from openai import OpenAI

from src.agents import MultiAgentState, AgentResult
from src.core import create_vectorstore, create_retrieval_grader


# ── Standalone (simple direct call) ─────────────────────────────

def search_books(question: str) -> str:
    """Search books and return a direct answer. Simple call, no graph state."""
    retriever = create_vectorstore()
    documents = retriever.invoke(question)

    grader = create_retrieval_grader()
    relevant_docs = []
    for doc in documents:
        score = grader.invoke({"question": question, "document": doc.page_content})
        if score.binary_score == "yes":
            relevant_docs.append(doc)

    if not relevant_docs:
        return "No relevant information found in your books."

    client = OpenAI()
    context = "\n\n---\n\n".join(d.page_content for d in relevant_docs)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a librarian. Using ONLY the provided book excerpts, "
                    "answer the user's question. Cite which passage you used. "
                    "If the excerpts don't contain the answer, say so honestly."
                ),
            },
            {
                "role": "user",
                "content": f"Book excerpts:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return resp.choices[0].message.content or "No answer could be generated."


# ── Coordinator pipeline node ───────────────────────────────────


def librarian_node(state: MultiAgentState) -> dict:
    """Search books, grade relevance, return findings."""
    query = state.get("translated_query") or state["query"]
    prior = state.get("agent_results", [])

    # ── Retrieve ────────────────────────────────────────────────
    retriever = create_vectorstore()            # returns retriever
    documents = retriever.invoke(query)

    # ── Grade relevance ─────────────────────────────────────────
    grader = create_retrieval_grader()
    relevant_docs = []
    for doc in documents:
        score = grader.invoke({
            "question": query,
            "document": doc.page_content,
        })
        if score.binary_score == "yes":
            relevant_docs.append(doc)

    # ── Build result ────────────────────────────────────────────
    if relevant_docs:
        # Direct SDK call to synthesise an answer from the relevant docs
        client = OpenAI()
        context = "\n\n---\n\n".join(d.page_content for d in relevant_docs)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a librarian. Using ONLY the provided book excerpts, "
                        "answer the user's question. Cite which passage you used. "
                        "If the excerpts don't contain the answer, say so honestly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Book excerpts:\n{context}\n\nQuestion: {query}",
                },
            ],
        )
        content = resp.choices[0].message.content or ""
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
        # If nothing found → flag for human confirmation (Option C / Q6)
        "needs_human_confirm": confidence == "none",
        "human_confirm_message": (
            "📚 Librarian found nothing in your books. "
            "Should I search the web instead?"
            if confidence == "none"
            else ""
        ),
    }
