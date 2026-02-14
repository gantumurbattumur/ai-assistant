"""Node functions for the LangGraph workflow"""
from langchain_core.documents import Document


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

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        grade = score.binary_score
        if grade == 'yes':
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
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
    # Re-write the question
    better_question = question_rewriter.invoke({"question": question})
    return {
        "question": better_question,
        "documents": documents
    }


def web_search(state, web_search_tool):
    """Perform web search to supplement documents"""
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join(d["content"] for d in docs)
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {
        "documents": documents,
        "question": question
    }


def generate(state, rag_chain):
    """Generate answer using RAG"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})

    return {
        "documents": documents,
        "question": question,
        "generation": generation
    }


def decide_to_generate(state):
    """
    Decides whether to generate an answer or re-generate a question
    
    Args:
        state: The current graph state
        
    Returns:
        str: Next node to call - "transform_query" or "generate"
    """
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    if web_search == "Yes":
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT. TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"
