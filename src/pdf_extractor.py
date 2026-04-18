"""
PDF Text Extraction Module
Extracts text from PDFs using native extraction (PyMuPDF) with OCR fallback (Tesseract).
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class DocumentPage:
    """A single page of extracted text."""
    page_number: int
    text: str
    method: str  # "native" or "ocr"


@dataclass
class Document:
    """A fully extracted PDF document."""
    filepath: str
    filename: str
    pages: list[DocumentPage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def num_pages(self) -> int:
        return len(self.pages)


class PDFExtractor:
    """Extract text from PDF files with native + OCR fallback."""

    def __init__(
        self,
        ocr_enabled: bool = True,
        ocr_language: str = "eng",
        min_text_length: int = 50,
    ):
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.min_text_length = min_text_length

    def extract(self, filepath: str) -> Document:
        """Extract text from a single PDF file."""
        filepath = str(Path(filepath).resolve())
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF not found: {filepath}")

        logger.info(f"Extracting text from: {filepath}")
        doc = fitz.open(filepath)

        # Extract metadata
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "num_pages": doc.page_count,
        }

        pages = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text")

            method = "native"
            # If native extraction yields little text, try OCR
            if len(text.strip()) < self.min_text_length and self.ocr_enabled:
                text = self._ocr_page(page)
                method = "ocr"

            pages.append(DocumentPage(
                page_number=page_num + 1,
                text=text,
                method=method,
            ))

        doc.close()

        return Document(
            filepath=filepath,
            filename=os.path.basename(filepath),
            pages=pages,
            metadata=metadata,
        )

    def _ocr_page(self, page) -> str:
        """OCR a single page by rendering to image and running Tesseract."""
        try:
            import pytesseract
            from PIL import Image
            import io

            # Render page to image at 300 DPI
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang=self.ocr_language)
            return text
        except ImportError:
            logger.warning("pytesseract or Pillow not installed. Skipping OCR.")
            return ""
        except Exception as e:
            logger.warning(f"OCR failed for page: {e}")
            return ""

    def extract_from_directory(
        self, directory: str, recursive: bool = False
    ) -> list[Document]:
        """Extract text from all PDFs in a directory."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(directory.glob(pattern))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []

        documents = []
        for pdf_path in pdf_files:
            try:
                doc = self.extract(str(pdf_path))
                documents.append(doc)
                logger.info(
                    f"  ✓ {doc.filename}: {doc.num_pages} pages, "
                    f"{len(doc.full_text)} chars"
                )
            except Exception as e:
                logger.error(f"  ✗ Failed to extract {pdf_path.name}: {e}")

        logger.info(f"Extracted {len(documents)} documents from {directory}")
        return documents
