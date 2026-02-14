from app.ingest.loaders import load_all_books
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)      

docs = load_all_books()
for doc in docs:
    print(doc[0].page_content[:100])