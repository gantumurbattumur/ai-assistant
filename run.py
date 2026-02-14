"""Main entry point for the Advanced RAG system"""
from pprint import pprint
from src.config import setup_environment
from src.graph.app import app


def main():
    """Run the Advanced RAG system"""
    # Setup environment variables
    setup_environment()
    
    # Run the workflow
    inputs = {"question": "What is the future of AI Engineering?"}
    
    print("\n=== Running Advanced RAG System ===\n")
    
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':")
        pprint("\n---\n")
    
    # Print final generation
    print("\n=== Final Answer ===\n")
    pprint(value["generation"])


if __name__ == "__main__":
    main()
