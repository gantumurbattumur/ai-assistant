"""Compile and build the LangGraph workflow"""
from langgraph.graph import END, START, StateGraph
from src.state import GraphState
from src.graph.nodes import (
    grade_documents,
    retrieve,
    transform_query,
    web_search,
    generate,
    decide_to_generate,
)
from src.tools import web_search_tool
from src.chains import create_rag_chain, create_retrieval_grader, create_question_rewriter
from src.retriever import create_vectorstore
from functools import partial


def create_workflow():
    """
    Create and compile the LangGraph workflow
    
    Returns:
        Compiled workflow app
    """
    # Initialize components
    retriever = create_vectorstore()
    rag_chain = create_rag_chain()
    retrieval_grader = create_retrieval_grader()
    question_rewriter = create_question_rewriter()
    
    # Create workflow
    workflow = StateGraph(GraphState)
    
    # Add nodes with partial functions to inject dependencies
    workflow.add_node("retrieve", partial(retrieve, retriever=retriever))
    workflow.add_node("grade_documents", partial(grade_documents, retrieval_grader=retrieval_grader))
    workflow.add_node("transform_query", partial(transform_query, question_rewriter=question_rewriter))
    workflow.add_node("generate", partial(generate, rag_chain=rag_chain))
    workflow.add_node("web_search", partial(web_search, web_search_tool=web_search_tool))
    
    # Build graph edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents", 
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


# Create the compiled app
app = create_workflow()
