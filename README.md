# Advanced RAG

An advanced Retrieval-Augmented Generation (RAG) system for processing and querying books in PDF and EPUB formats using LangGraph, with adaptive query rewriting and web search fallback.

## 🏗️ Project Structure

```
advanced-rag/
├── app/
│   └── ingest/
│       ├── loaders.py          # Document loaders for PDF and EPUB
│       └── chunking.py         # Text chunking utilities
├── src/
│   ├── config.py               # Environment configuration
│   ├── state.py                # Graph state definition
│   ├── chains.py               # LLM chains (grader, generator, rewriter)
│   ├── tools.py                # External tools (web search)
│   ├── retriever.py            # Vector store and retriever setup
│   └── graph/
│       ├── nodes.py            # LangGraph node functions
│       └── app.py              # Graph compilation and workflow
├── data/
│   └── books/                  # Book files (PDF, EPUB)
├── main.ipynb                  # Jupyter notebook for experimentation
├── run.py                      # Main entry point
├── pyproject.toml              # Project dependencies
└── README.md
```

## 🚀 Features

- **Document Processing**: Load and process PDF and EPUB files
- **Intelligent Chunking**: Split documents using tiktoken-based text splitter
- **Semantic Search**: Vector-based retrieval using ChromaDB and OpenAI embeddings
- **Document Grading**: LLM-powered relevance scoring for retrieved documents
- **Adaptive Query**: Automatically rewrites queries for better web search results
- **Web Search Fallback**: Uses Tavily API when retrieved documents are insufficient
- **LangGraph Workflow**: Structured, debuggable multi-step RAG pipeline

## 📦 Installation

1. **Clone the repository**:
   ```bash
   cd advanced-rag
   ```

2. **Install dependencies with uv**:
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export TAVILY_API_KEY="your-tavily-api-key"
   ```

## 🎯 Usage

### Run the full RAG pipeline:

```bash
uv run python run.py
```

### Use in Jupyter Notebook:

```bash
jupyter notebook main.ipynb
```

### Programmatic usage:

```python
from src.config import setup_environment
from src.graph.app import app

# Setup environment
setup_environment()

# Run query
inputs = {"question": "What is the future of AI Engineering?"}
for output in app.stream(inputs):
    print(output)
```

## 🔧 How It Works

The RAG system follows this workflow:

1. **Retrieve**: Fetch relevant documents from vector store
2. **Grade**: Score documents for relevance to the question
3. **Decision**: 
   - If documents are relevant → Generate answer
   - If documents are not relevant → Transform query
4. **Transform Query** (if needed): Rewrite question for web search
5. **Web Search** (if needed): Fetch additional context from web
6. **Generate**: Create final answer using retrieved context

## 📚 Adding Books

Place your PDF or EPUB files in the `data/books/` directory. The system will automatically load and process them.

## 🛠️ Development

### Install dev dependencies:

```bash
uv sync --dev
```

### Format code:

```bash
uv run black .
```

### Lint code:

```bash
uv run ruff check .
```

## 📝 Dependencies

- **LangChain**: LLM orchestration and chains
- **LangGraph**: Workflow and state management
- **OpenAI**: LLM and embeddings
- **ChromaDB**: Vector storage
- **Tavily**: Web search API
- **tiktoken**: Token counting and chunking

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License

