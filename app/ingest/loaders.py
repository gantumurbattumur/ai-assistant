from langchain_community.document_loaders import PyMuPDFLoader
from pprint import pprint
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_core.documents import Document

FILE_PATH = "data/books/"

# Load a single PDF file 
def load_pdf(file_name):
    loader = PyMuPDFLoader(FILE_PATH + file_name)
    return loader.load()

# Load a single EPUB file (simple implementation without heavy dependencies)
def load_epub(file_name):
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

# Load multiple PDF files and return a list of documents
def load_files(file_names):
    documents = []
    for file_name in file_names:
        documents.append(load_pdf(file_name))
    return documents

# Load multiple EPUB files and return a list of documents
def load_epub_files(file_names):
    documents = []
    for file_name in file_names:
        documents.append(load_epub(file_name))
    return documents

# Load all books (pdf, epub) from the data/books directory and return a list of documents
def load_all_books():
    import os
    documents = []
    for file_name in os.listdir(FILE_PATH):
        if file_name.endswith(".pdf"):
            documents.append(load_pdf(file_name))
        elif file_name.endswith(".epub"):
            documents.append(load_epub(file_name))
    return documents

