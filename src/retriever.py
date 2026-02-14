"""Retriever setup and initialization"""
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingest.loaders import load_all_books


def create_vectorstore():
    """
    Load documents, split them, and create a vectorstore with retriever
    
    Returns:
        retriever: Vector store retriever
    """
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
    
    retriever = vectorstore.as_retriever()
    return retriever
