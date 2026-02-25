"""Multi-agent LangGraph workflow.

The Coordinator produces a plan (ordered list of agent names).
The Dispatcher node runs each agent in sequence, advancing current_step.
Between steps it checks for human confirmation needs and short-circuit flags.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents import MultiAgentState
from src.agents.coordinator import coordinator_node
from src.agents.librarian import librarian_node
from src.agents.researcher import researcher_node
from src.agents.translator import translator_node
from src.agents.summarizer import summarizer_node
from src.agents.critic import critic_node
from src.tasks.macos_agent import task_agent_node
from src.config import MODEL_NAME, get_openai_client

# ── Agent registry ──────────────────────────────────────────────
AGENT_NODES = {
    "librarian": librarian_node,
    "researcher": researcher_node,
    "translator": translator_node,
    "summarizer": summarizer_node,
    "critic": critic_node,
    "task_agent": task_agent_node,
}


# ── Dispatcher: runs the current agent in the plan ──────────────

def dispatcher_node(state: MultiAgentState) -> dict:
    """Execute the current agent in the plan."""
    plan = state.get("plan", [])
    step = state.get("current_step", 0)

    if step >= len(plan):
        return {"should_stop": True}

    agent_name = plan[step]
    agent_fn = AGENT_NODES.get(agent_name)

    if agent_fn is None:
        # Unknown agent → skip
        return {"current_step": step + 1}

    # Run the agent
    result = agent_fn(state)

    # Advance the step counter
    result["current_step"] = step + 1

    return result


# ── Simple answer node (empty plan → direct LLM answer) ────────

def simple_answer_node(state: MultiAgentState) -> dict:
    """For simple queries where no specialist agent is needed."""
    query = state["query"]

    resp = get_openai_client().chat.completions.create(
        model=MODEL_NAME,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful, concise AI assistant. Answer clearly.",
            },
            {"role": "user", "content": query},
        ],
    )
    content = resp.choices[0].message.content or ""

    return {
        "response": content,
        "agents_used": ["🤖 Direct (simple query)"],
        "should_stop": True,
    }


# ── Finalizer: if no response set yet, pick from agent_results ──

def finalizer_node(state: MultiAgentState) -> dict:
    """Ensure there's a response. Pick from agent_results if not set."""
    if state.get("response"):
        return {}

    results = state.get("agent_results", [])
    # Pick the last content-producing agent's output
    for r in reversed(results):
        if r["content"]:
            return {"response": r["content"]}

    return {"response": "I wasn't able to find an answer. Please try rephrasing your question."}


# ── Edge decisions ──────────────────────────────────────────────

def after_coordinator(state: MultiAgentState) -> str:
    """After coordinator: go to dispatcher if there's a plan, else simple answer."""
    plan = state.get("plan", [])
    if not plan:
        return "simple_answer"
    return "dispatcher"


def after_dispatcher(state: MultiAgentState) -> str:
    """After dispatcher: continue, ask human, or finalize."""
    # Human confirmation needed? (Option C / Q6)
    if state.get("needs_human_confirm", False):
        return "human_check"

    # Short-circuit?
    if state.get("should_stop", False):
        return "finalizer"

    # More agents to run?
    plan = state.get("plan", [])
    step = state.get("current_step", 0)
    if step < len(plan):
        return "dispatcher"

    # All done
    return "finalizer"


def human_check_node(state: MultiAgentState) -> dict:
    """Pause and ask the user for confirmation before continuing.

    Uses langgraph.types.interrupt so the graph genuinely suspends.
    The CLI resumes the graph with the user's answer via Command(resume=...).
    If the user declines, should_stop is set to True to skip remaining agents.
    """
    from langgraph.types import interrupt

    msg = state.get("human_confirm_message", "Continue?")
    confirmed = interrupt({"message": msg, "options": ["yes", "no"]})

    if str(confirmed).strip().lower() in ("yes", "y", ""):
        return {
            "needs_human_confirm": False,
            "human_confirm_message": "",
        }
    else:
        return {
            "needs_human_confirm": False,
            "human_confirm_message": "",
            "should_stop": True,
        }


def after_human_check(state: MultiAgentState) -> str:
    """After human confirmed → continue dispatching or finalize."""
    if state.get("should_stop", False):
        return "finalizer"

    plan = state.get("plan", [])
    step = state.get("current_step", 0)
    if step < len(plan):
        return "dispatcher"

    return "finalizer"


# ── Build the graph ─────────────────────────────────────────────

def create_multi_agent_graph():
    """Create and compile the multi-agent LangGraph workflow.

    Uses an in-memory checkpointer so that interrupt() in human_check_node
    can genuinely pause the graph and resume after user confirmation.
    """
    from langgraph.checkpoint.memory import MemorySaver

    workflow = StateGraph(MultiAgentState)

    # Add nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("dispatcher", dispatcher_node)
    workflow.add_node("simple_answer", simple_answer_node)
    workflow.add_node("human_check", human_check_node)
    workflow.add_node("finalizer", finalizer_node)

    # Edges
    workflow.add_edge(START, "coordinator")

    workflow.add_conditional_edges(
        "coordinator",
        after_coordinator,
        {
            "dispatcher": "dispatcher",
            "simple_answer": "simple_answer",
        },
    )

    workflow.add_conditional_edges(
        "dispatcher",
        after_dispatcher,
        {
            "dispatcher": "dispatcher",
            "human_check": "human_check",
            "finalizer": "finalizer",
        },
    )

    workflow.add_conditional_edges(
        "human_check",
        after_human_check,
        {
            "dispatcher": "dispatcher",
            "finalizer": "finalizer",
        },
    )

    workflow.add_edge("simple_answer", END)
    workflow.add_edge("finalizer", END)

    return workflow.compile(checkpointer=MemorySaver())
