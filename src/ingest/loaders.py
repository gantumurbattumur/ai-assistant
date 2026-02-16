"""Document loaders for PDF and EPUB files"""
import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

FILE_PATH = "data/books/"


def load_pdf(file_name):
    """Load a single PDF file and return list of documents"""
    loader = PyMuPDFLoader(FILE_PATH + file_name)
    return loader.load()


def load_epub(file_name):
    """Load a single EPUB file and return list of documents"""
    book = epub.read_epub(FILE_PATH + file_name)
    documents = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content()
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text()
            
            documents.append(Document(
                page_content=text,
                metadata={"source": file_name}
            ))
    
    return documents


def load_all_books():
    """Load all PDF and EPUB files from data/books directory"""
    documents = []
    
    if not os.path.exists(FILE_PATH):
        print(f"Warning: Books directory not found at {FILE_PATH}")
        return []
    
    for file_name in os.listdir(FILE_PATH):
        if file_name.endswith(".pdf"):
            documents.append(load_pdf(file_name))
        elif file_name.endswith(".epub"):
            documents.append(load_epub(file_name))
    
    return documents


