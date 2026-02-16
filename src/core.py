"""Core RAG system components - chains, tools, and configuration"""
import getpass
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import TypedDict


# ========== Configuration ==========

def setup_environment():
    """Setup required environment variables"""
    for key in ["OPENAI_API_KEY", "TAVILY_API_KEY"]:
        if key not in os.environ:
            os.environ[key] = getpass.getpass(f"{key}:")


# ========== State Definition ==========

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


# ========== Data Models ==========

class GradeDocuments(BaseModel):
    """Binary score for relevance check for retrieved documents"""
    binary_score: str = Field(
        description="Documents are relevant to the query, 'yes' or 'no'"
    )


# ========== Chains ==========

def create_retrieval_grader():
    """Create a chain to grade document relevance"""
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    system = """You are a grader for the relevance of retrieved documents to the question. 
    If the document contains keyword(s) or semantic meaning to the question, grade it as relevant. 
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ('system', system),
        ('human', "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ])
    
    return grade_prompt | structured_llm_grader


def create_rag_chain():
    """Create the main RAG chain for generation"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant. Use the provided context to answer the question. "
         "If you don't know the answer based on the context, say you don't know."),
        ("human", 
         "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])
    
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    return prompt | llm | StrOutputParser()


def create_question_rewriter():
    """Create a chain to rewrite questions for better web search"""
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    
    system = """You are a question re-writer. Your task is to re-write the user's question 
    to a better version that is optimized for web search. Look at the input and try to reason 
    about the underlying semantic intent / meaning."""
    
    re_write_prompt = ChatPromptTemplate.from_messages([
        ('system', system),
        ('human', "Here is the initial question:\n\n{question}\n\nFormulate an improved question."),
    ])
    
    return re_write_prompt | llm | StrOutputParser()


# ========== Tools ==========

web_search_tool = TavilySearchResults(k=3)


# ========== Retriever Setup ==========

def create_vectorstore():
    """
    Load documents, split them, and create a vectorstore with retriever
    
    Returns:
        retriever: Vector store retriever
    """
    from src.ingest.loaders import load_all_books
    
    print("Loading books...")
    docs = load_all_books()
    print(f"Loaded {len(docs)} documents")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, 
        chunk_overlap=0
    )
    
    doc_splits = text_splitter.create_documents(
        [doc.page_content for sublist in docs for doc in sublist]
    )
    print(f"Split into {len(doc_splits)} chunks")
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    
    return vectorstore.as_retriever()
