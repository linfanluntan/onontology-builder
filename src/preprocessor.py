"""
Text Preprocessing Module
Cleans and segments extracted PDF text for downstream NLP tasks.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TextSegment:
    """A logical segment of text (section, paragraph, etc.)."""
    text: str
    heading: str = ""
    segment_type: str = "paragraph"  # heading, paragraph, list, table
    source_doc: str = ""
    page_number: int = 0


class TextPreprocessor:
    """Clean and segment raw PDF text."""

    def __init__(
        self,
        remove_headers_footers: bool = True,
        normalize_whitespace: bool = True,
        fix_hyphenation: bool = True,
        min_segment_length: int = 20,
    ):
        self.remove_headers_footers = remove_headers_footers
        self.normalize_whitespace = normalize_whitespace
        self.fix_hyphenation = fix_hyphenation
        self.min_segment_length = min_segment_length

    def preprocess(self, text: str, source_doc: str = "") -> list[TextSegment]:
        """Full preprocessing pipeline on raw text."""
        text = self._clean(text)
        segments = self._segment(text, source_doc)
        return segments

    def _clean(self, text: str) -> str:
        """Clean raw extracted text."""
        if self.fix_hyphenation:
            # Rejoin hyphenated words at line breaks
            text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        if self.normalize_whitespace:
            # Collapse multiple spaces/tabs to single space
            text = re.sub(r"[ \t]+", " ", text)
            # Collapse 3+ newlines to 2
            text = re.sub(r"\n{3,}", "\n\n", text)

        if self.remove_headers_footers:
            # Remove common page number patterns
            text = re.sub(r"\n\s*-?\s*\d+\s*-?\s*\n", "\n", text)
            text = re.sub(r"\n\s*Page\s+\d+\s*(of\s+\d+)?\s*\n", "\n", text, flags=re.I)

        # Remove non-printable characters (keep newlines and tabs)
        text = re.sub(r"[^\S\n\t]+", " ", text)

        return text.strip()

    def _segment(self, text: str, source_doc: str = "") -> list[TextSegment]:
        """Split text into logical segments based on structure."""
        segments = []

        # Split on double newlines (paragraph boundaries)
        raw_segments = re.split(r"\n{2,}", text)

        current_heading = ""
        for raw in raw_segments:
            raw = raw.strip()
            if len(raw) < self.min_segment_length:
                continue

            # Detect headings: short lines, possibly uppercase or numbered
            if self._is_heading(raw):
                current_heading = raw
                segments.append(TextSegment(
                    text=raw,
                    heading=raw,
                    segment_type="heading",
                    source_doc=source_doc,
                ))
            else:
                segments.append(TextSegment(
                    text=raw,
                    heading=current_heading,
                    segment_type="paragraph",
                    source_doc=source_doc,
                ))

        return segments

    def _is_heading(self, text: str) -> bool:
        """Heuristic: detect if a text block is likely a heading."""
        lines = text.strip().split("\n")
        if len(lines) > 2:
            return False
        first_line = lines[0].strip()
        if len(first_line) > 120:
            return False
        # All caps
        if first_line.isupper() and len(first_line) > 3:
            return True
        # Numbered heading: "1.2 Introduction" or "Chapter 3"
        if re.match(r"^(\d+\.?\d*\.?\d*)\s+\w", first_line):
            return True
        if re.match(r"^(Chapter|Section|Part)\s+\d", first_line, re.I):
            return True
        return False

    def preprocess_documents(self, documents) -> list[TextSegment]:
        """Preprocess multiple Document objects."""
        all_segments = []
        for doc in documents:
            segments = self.preprocess(doc.full_text, source_doc=doc.filename)
            all_segments.extend(segments)
            logger.info(f"  {doc.filename}: {len(segments)} segments")
        return all_segments
