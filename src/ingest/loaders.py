"""Document loaders using Docling for multimodal parsing (PDF, EPUB, DOCX, PPTX, images)."""
import os
from pathlib import Path
from langchain_core.documents import Document

# Resolved at import time relative to this file, so it works regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FILE_PATH = str(_PROJECT_ROOT / "data" / "books") + "/"

SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".docx", ".pptx", ".xlsx", ".html", ".md"}


def _build_docling_converter():
    """Build and return a configured Docling DocumentConverter."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,                   # OCR for scanned pages
        do_table_structure=True,       # Extract tables as structured Markdown
        generate_picture_images=True,  # Capture embedded images
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


# Module-level cache — Docling loads ML models on first use, so only build once.
_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        _converter = _build_docling_converter()
    return _converter


# ---------------------------------------------------------------------------
# Core Docling parser
# ---------------------------------------------------------------------------

def _docs_from_docling(file_path: str, file_name: str) -> list[Document]:
    """
    Parse a single file with Docling and return a flat list of LangChain Documents.

    Each Document carries metadata:
        content_type : "text" | "heading" | "table" | "image_description"
        source       : original filename
        page         : 1-based page number (or None)
    """
    from docling_core.types.doc.document import TextItem, TableItem, PictureItem, SectionHeaderItem, DocItem

    converter = _get_converter()
    result = converter.convert(file_path)
    doc = result.document

    documents: list[Document] = []

    for element, _level in doc.iterate_items():
        # Cast to DocItem to access typed fields (iterate_items yields NodeItem at type level)
        item: DocItem = element  # type: ignore[assignment]

        # --- page number from provenance ---
        page: int | None = None
        if item.prov:
            page = item.prov[0].page_no  # 1-based

        base_meta = {"source": file_name, "page": page}

        # ----- Tables → Markdown ------------------------------------------------
        if isinstance(item, TableItem):
            try:
                content = item.export_to_markdown()
            except Exception:
                content = item.caption_text(doc) if item.captions else ""
            if content.strip():
                documents.append(Document(
                    page_content=content,
                    metadata={**base_meta, "content_type": "table"},
                ))
            continue

        # ----- Pictures → caption text ------------------------------------------
        if isinstance(item, PictureItem):
            caption = item.caption_text(doc) if item.captions else ""
            if caption.strip():
                documents.append(Document(
                    page_content=f"[Image] {caption}",
                    metadata={**base_meta, "content_type": "image_description"},
                ))
            continue

        # ----- Section headings -------------------------------------------------
        if isinstance(item, SectionHeaderItem):
            if item.text.strip():
                documents.append(Document(
                    page_content=item.text,
                    metadata={**base_meta, "content_type": "heading"},
                ))
            continue

        # ----- Plain text (TextItem, ListItem, …) --------------------------------
        if isinstance(item, TextItem):
            if item.text.strip():
                documents.append(Document(
                    page_content=item.text,
                    metadata={**base_meta, "content_type": "text"},
                ))

    return documents


# ---------------------------------------------------------------------------
# EPUB fallback (Docling EPUB support is experimental)
# ---------------------------------------------------------------------------

def _docs_from_epub(file_path: str, file_name: str) -> list[Document]:
    """Fallback EPUB loader using ebooklib + BeautifulSoup."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(file_path)
    documents: list[Document] = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n").strip()
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file_name, "page": None, "content_type": "text"},
                ))

    return documents


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_file(file_path: str) -> list[Document]:
    """
    Load a single file and return a flat list of LangChain Documents.

    Docling handles PDF / DOCX / PPTX / XLSX / HTML / MD natively.
    EPUB falls back to the ebooklib-based loader.
    """
    p = Path(file_path)
    ext = p.suffix.lower()

    if ext == ".epub":
        return _docs_from_epub(str(p), p.name)
    elif ext in SUPPORTED_EXTENSIONS:
        return _docs_from_docling(str(p), p.name)
    else:
        print(f"  ⚠️  Skipping unsupported format: {p.name}")
        return []


def load_all_books() -> list[Document]:
    """
    Load all supported documents from data/books/ and return a **flat** list
    of LangChain Documents (one element = one Document).

    Each Document has metadata:
        source, page, content_type  ∈ {"text", "heading", "table", "image_description"}
    """
    if not os.path.exists(FILE_PATH):
        print(f"Warning: Books directory not found at {FILE_PATH}")
        return []

    all_docs: list[Document] = []

    for file_name in sorted(os.listdir(FILE_PATH)):
        ext = Path(file_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        file_path = os.path.join(FILE_PATH, file_name)
        print(f"  📄 Parsing {file_name} …")
        try:
            docs = load_file(file_path)
            print(f"     → {len(docs)} elements")
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ❌ Failed to load {file_name}: {e}")

    return all_docs


