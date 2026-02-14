"""Graph state definition"""
from typing import List
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of the graph

    Attributes:
        question: User question
        generation: LLM generation
        web_search: Whether to search, 'yes' or 'no'
        documents: List of documents retrieved
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]
