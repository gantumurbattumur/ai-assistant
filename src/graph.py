"""LangGraph workflow nodes and graph compilation"""
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from functools import partial
from src.core import (
    GraphState,
    create_rag_chain,
    create_retrieval_grader,
    create_question_rewriter,
    create_vectorstore,
    get_web_search_tool,
)


# ========== Node Functions ==========

def retrieve(state, retriever):
    """Retrieve documents based on the question"""
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state, retrieval_grader):
    """Grade document relevance to the question"""
    print("---CHECK DOCUMENTS RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    web_search = "No"
    for doc in documents:
        score = retrieval_grader.invoke({
            "question": question, 
            "document": doc.page_content
        })
        grade = score.binary_score
        if grade == 'yes':
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
    
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search
    }


def transform_query(state, question_rewriter):
    """Optimize the query for web search"""
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question, "documents": documents}


def web_search(state, web_search_tool):
    """Perform web search to supplement documents"""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join(d["content"] for d in docs)
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    
    return {"documents": documents, "question": question}


def generate(state, rag_chain):
    """Generate answer using RAG"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    generation = rag_chain.invoke({
        "context": documents, 
        "question": question
    })
    
    return {
        "documents": documents,
        "question": question,
        "generation": generation
    }


def decide_to_generate(state):
    """Decide whether to generate an answer or transform query for web search"""
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    
    if web_search == "Yes":
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT. TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


# ========== Graph Compilation ==========

def create_graph():
    """Create and compile the LangGraph workflow"""
    # Initialize components
    retriever = create_vectorstore()
    rag_chain = create_rag_chain()
    retrieval_grader = create_retrieval_grader()
    question_rewriter = create_question_rewriter()
    web_search_tool = get_web_search_tool()
    
    # Create workflow
    workflow = StateGraph(GraphState)
    
    # Add nodes with dependencies injected
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
