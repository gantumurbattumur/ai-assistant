# AI Assistant

AI Assitant with an advanced Retrieval-Augmented Generation (RAG) system that processes and queries books in PDF and EPUB formats. Built with LangGraph for adaptive query rewriting and web search fallback capabilities.

## 🏗️ Architecture Overview

The system is organized into a clean, modular structure:

```
advanced-rag/
├── src/
│   ├── cli.py                  # Command-line interface (ai commands)
│   ├── core.py                 # RAG components (chains, tools, config)
│   ├── graph.py                # LangGraph workflow definition
│   └── ingest/
│       └── loaders.py          # Document loaders (PDF/EPUB)
├── data/
│   └── books/                  # Book collection (PDFs, EPUBs)
├── chroma_db/                  # Persistent vector index
├── main.ipynb                  # Jupyter notebook for experimentation
└── pyproject.toml              # Project dependencies & metadata
```

**Core Components:**
- **`cli.py`**: User interface with commands (ask, rag ask, search, translate, etc.)
- **`core.py`**: Configuration, state management, RAG chains, and tools
- **`graph.py`**: LangGraph nodes and workflow compilation

## 🚀 Features

### **Intelligent Document Processing**
- Loads PDF and EPUB files from the book collection
- Splits documents into 250-token chunks using tiktoken encoder
- Creates persistent vector embeddings using OpenAI's embedding model
- Stores in ChromaDB for fast semantic search

### **Advanced RAG Pipeline**
- **Semantic Search**: Vector-based retrieval using cosine similarity
- **Document Grading**: LLM-powered relevance scoring for retrieved chunks
- **Adaptive Query Rewriting**: Automatically rewrites questions for better web search
- **Web Search Fallback**: Uses Tavily API when book content is insufficient
- **Persistent Index**: Built once, reused across queries (20-30× faster after first run)

### **Multi-Modal AI Assistant**
- **RAG Q&A**: Ask questions about your personal book collection
- **Quick Ask**: Direct LLM queries without document retrieval
- **Web Search**: Current information via Tavily API
- **Summarization**: Summarize text or web pages
- **Translation**: Translate text to any language
- **Utilities**: Jokes, system info, version management

## 🔧 How It Works

### **1. Document Ingestion & Indexing**

When you first run a RAG query:
1. **Load Documents**: Scans `data/books/` for PDFs and EPUBs
2. **Extract Text**: Uses PyMuPDF for PDFs, BeautifulSoup for EPUBs
3. **Chunk Content**: Splits into 250-token chunks with tiktoken (optimized for context)
4. **Generate Embeddings**: OpenAI's `text-embedding-ada-002` creates vector representations
5. **Store in ChromaDB**: Saves to `chroma_db/` directory for persistence
6. **Build Index**: Creates semantic search index for fast retrieval

**Performance**: First run takes ~45-60 seconds, subsequent runs ~2-3 seconds (loads existing index).

### **2. RAG Workflow (LangGraph Pipeline)**

The system uses a **LangGraph state machine** with conditional branching:

```
START
  ↓
[Retrieve] ────────────> Fetch top-k documents from vector store
  ↓
[Grade Documents] ─────> LLM scores each doc for relevance (yes/no)
  ↓
[Decision Node] ───────> Are documents sufficient?
  ├─ YES ──────────────> [Generate Answer] ─> END
  └─ NO ───────────────> [Transform Query]
                            ↓
                         [Web Search] ────> Fetch additional context
                            ↓
                         [Generate Answer] ─> END
```

**Node Details:**

**Retrieve Node:**
- Queries ChromaDB vector store with question embedding
- Returns top 4-5 most similar document chunks
- Uses cosine similarity for ranking

**Grade Documents Node:**
- For each retrieved document, asks GPT-3.5: "Is this relevant to the question?"
- Uses structured output (Pydantic model) for binary yes/no
- Filters out irrelevant documents
- If any docs are irrelevant, triggers web search path

**Transform Query Node:**
- Rewrites the original question for better web search results
- Uses GPT-3.5 with prompt: "Optimize this question for web search"
- Example: "What's RAG?" → "What is Retrieval Augmented Generation in AI?"

**Web Search Node:**
- Calls Tavily API with transformed query
- Fetches top 3 web results
- Appends web content to document list

**Generate Answer Node:**
- Combines all documents (book chunks + web results)
- Uses GPT-3.5 with RAG prompt: "Answer based on this context"
- Streams response back to user

### **3. State Management**

Uses TypedDict for type-safe state passing between nodes:

```python
class GraphState(TypedDict):
    question: str          # User's question
    documents: List[str]   # Retrieved + web search docs
    web_search: str        # "Yes" or "No" flag
    generation: str        # Final answer
```

Each node receives current state, performs its operation, and returns updated state. LangGraph handles routing and execution.

### **4. Persistent Vector Storage**

**First Run (Build Index):**
- Loads all books from `data/books/`
- Generates embeddings for ~15,000 chunks (typical library)
- Costs ~$0.50-$1.00 in OpenAI API calls
- Saves to `chroma_db/` directory (15-50 MB)

**Subsequent Runs (Load Index):**
- Checks if `chroma_db/` exists
- Loads existing embeddings from disk
- No API calls, no cost
- ~2-3 second startup time

**Rebuilding Index:**
- Use `ai rag rebuild` when adding new books
- Deletes old index and regenerates from scratch

### **5. CLI Command System**

Built with Typer and Rich for beautiful terminal UX:

**Command Structure:**
- `ai ask "..."` - Direct LLM query (no RAG)
- `ai rag ask "..."` - RAG pipeline with books
- `ai rag status` - Show index statistics
- `ai rag rebuild` - Rebuild vector index
- `ai search "..."` - Web search via Tavily
- `ai summarize "..."` - Summarize text/URL
- `ai translate "..." --to French` - Translation
- `ai joke` - Random joke (no API key needed)
- `ai info` - System information

**Features:**
- Streaming output (see workflow progress in real-time)
- Verbose mode (shows intermediate states)
- Rich markdown rendering
- Progress spinners and panels
- Error handling with helpful messages

## 📊 Technical Stack

### **Core Dependencies**
- **LangChain**: LLM orchestration, chains, and tools
- **LangGraph**: State machine workflow management
- **OpenAI**: GPT-3.5-turbo (LLM), text-embedding-ada-002 (embeddings)
- **ChromaDB**: Persistent vector database
- **Tavily**: Web search API with AI-optimized results


### **CLI & UX**
- **Typer**: Command-line interface framework
- **Rich**: Beautiful terminal formatting (markdown, panels, progress bars)
- **httpx**: Modern HTTP client for API calls

## 🔑 Key Design Decisions

### **Why Persistent Vector Storage?**
- Avoids re-embedding same documents (saves $$ and time)
- 20-30× faster startup after initial build
- Supports large book collections without performance degradation

### **Why 250-Token Chunks?**
- Balances context vs. precision
- Fits within LLM context windows comfortably
- Tested across various document types for optimal retrieval

### **Why LangGraph Over Simple Chains?**
- Visibility into workflow (can see each step)
- Conditional logic (adaptive web search)
- Debuggable (can inspect state at each node)
- Extensible (easy to add new nodes)

### **Why Hybrid Approach (RAG + Web)?**
- Books provide deep, authoritative knowledge
- Web provides current, up-to-date information
- Combination gives comprehensive answers

### **Why CLI First?**
- Fast iteration during development
- Low resource overhead
- Easy to script and automate
- Can be wrapped with web UI later

## � Performance Characteristics

### **Query Latency**
- **Cold start (first query)**: 45-60 seconds (builds index)
- **Warm queries**: 2-8 seconds depending on complexity
  - Vector search: ~200ms
  - LLM grading: ~1-2s
  - Web search (if triggered): ~1-2s
  - Answer generation: ~2-3s

## 📝 Notes

This is a personal AI assistant project focused on:
- Learning and experimenting with RAG techniques
- Building a useful tool for querying personal book collections
- Exploring LangChain and LangGraph capabilities
- Testing different prompt engineering approaches

Not intended for public distribution or production deployment.

