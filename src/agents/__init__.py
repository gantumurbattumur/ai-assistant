"""Multi-agent system — state definitions and shared types"""
from typing import List, Optional
from typing_extensions import TypedDict


class AgentResult(TypedDict):
    """Output from a single agent"""
    agent: str
    content: str
    sources: List[str]
    confidence: str  # "high", "medium", "low", "none"


class MultiAgentState(TypedDict):
    """Shared state flowing through the multi-agent graph"""
    # ---- input (required) ----
    query: str                        # original user query

    # ---- coordinator plan ----
    language: str                     # detected language of query ("en", "other")
    translated_query: str             # English version of query (if non-English)
    plan: List[str]                   # ordered list of agents to run
    plan_reasoning: str               # why coordinator chose this plan
    current_step: int                 # index into plan[]

    # ---- accumulated context (Option A: agents see prior results) ----
    agent_results: List[AgentResult]  # each agent appends its output here

    # ---- final output ----
    response: str                     # final answer shown to user
    agents_used: List[str]            # display: which agents ran

    # ---- control flags ----
    needs_human_confirm: bool         # True → ask user before continuing
    human_confirm_message: str        # what to show user
    should_stop: bool                 # True → short-circuit, skip remaining agents
