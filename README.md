# Advanced RAG

An advanced Retrieval-Augmented Generation (RAG) system for processing and querying books in PDF and EPUB formats.

## Features

- Load and process PDF and EPUB files
- Text chunking and splitting
- Document ingestion pipeline

## Installation

```bash
uv sync
```

## Usage

```python
from app.ingest.loaders import load_all_books

# Load all books from data/books directory
documents = load_all_books()
```
