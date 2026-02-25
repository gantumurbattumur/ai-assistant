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
from pathlib import Path
from dotenv import load_dotenv

# ========== Configuration ==========

def setup_environment():
    """Load environment variables from .env file.

    Search order:
      1. Current working directory (.env)
      2. Project root relative to this file (local dev)
      3. ~/.config/ai-assistant/.env  (recommended for installed tool)
      4. ~/.env  (home directory fallback)
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    candidates = [
        Path.cwd() / ".env",
        project_root / ".env",
        Path.home() / ".config" / "ai-assistant" / ".env",
        Path.home() / ".env",
    ]

    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate)
            break
    else:
        load_dotenv()  # let python-dotenv search by itself as last resort

    # Verify required keys are present
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found. Please create a .env file in the project root with:\n"
            "OPENAI_API_KEY=your-key-here"
        )
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY not found. Web search features may not work.")


# ========== State Definition ==========

class GraphState(TypedDict):
    """
    Represents the state of the graph

    Attributes:
        question: User question
        generation: LLM generation
        web_search: Whether to search, 'Yes' or 'No'
        documents: List of documents retrieved
        irrelevant_count: Number of irrelevant docs found during grading
        total_retrieved: Total docs retrieved before filtering
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]
    irrelevant_count: int
    total_retrieved: int


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

def get_web_search_tool():
    """Get web search tool (lazy initialization)"""
    from langchain_community.tools.tavily_search import TavilySearchResults
    return TavilySearchResults(k=3)


# ========== Retriever Setup ==========

# Module-level cache — avoids reloading the vectorstore on every call.
_retriever = None


def create_vectorstore(force_rebuild: bool = False):
    """
    Load or create a persistent vectorstore.

    Uses a module-level cache so subsequent calls return the same retriever
    without reloading Chroma or re-embedding documents.

    Args:
        force_rebuild: If True, drop the cache and rebuild the index from scratch.

    Returns:
        retriever: VectorStore retriever backed by the Chroma persistent store.
    """
    global _retriever

    # Return cached retriever unless a rebuild is requested
    if _retriever is not None and not force_rebuild:
        return _retriever

    from src.ingest.loaders import load_all_books

    # Persistent storage location
    project_root = Path(__file__).parent.parent
    persist_dir = project_root / "chroma_db"

    if force_rebuild:
        # Clear the in-memory cache so we rebuild cleanly
        _retriever = None

    # Check if index already exists on disk
    if persist_dir.exists() and not force_rebuild:
        print("Loading existing vectorstore...")
        vectorstore = Chroma(
            collection_name="rag-chroma",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=str(persist_dir),
        )
        print(f"✅ Loaded {vectorstore._collection.count()} existing chunks")
        _retriever = vectorstore.as_retriever()
        return _retriever

    # Build from scratch
    print("Building vectorstore from documents...")
    docs = load_all_books()

    if not docs:
        raise ValueError("No documents found. Add PDF/EPUB files to data/books/")

    print(f"Loaded {len(docs)} elements")

    # Split text/heading chunks; keep tables and image_descriptions as-is
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=0,
    )

    splittable = [d for d in docs if d.metadata.get("content_type") in ("text", "heading", None)]
    preserved  = [d for d in docs if d.metadata.get("content_type") in ("table", "image_description")]

    split_chunks = text_splitter.split_documents(splittable)
    doc_splits   = split_chunks + preserved

    print(f"Split into {len(doc_splits)} chunks ({len(preserved)} tables/images kept whole)")
    print("Generating embeddings (this may take a minute)...")

    # Create persistent vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
        persist_directory=str(persist_dir),
    )

    print(f"✅ Vectorstore saved to {persist_dir}")
    _retriever = vectorstore.as_retriever()
    return _retriever
