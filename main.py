"""
Main script for advanced RAG system
"""
from app.ingest.loaders import load_all_books
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    print("Loading books...")
    docs = load_all_books()
    print(f"Loaded {len(docs)} documents")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Show first document preview
    if docs:
        print("\nFirst document preview:")
        print(f"Source: {docs[0][0].metadata.get('source', 'Unknown')}")
        print(f"Content preview: {docs[0][0].page_content[:200]}...")
    
if __name__ == "__main__":
    main()
