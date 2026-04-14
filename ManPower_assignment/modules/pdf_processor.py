"""
pdf_processor.py
Handles PDF loading, validation, page-limit enforcement, and hash generation.
"""

import io
import hashlib
import pdfplumber


MAX_PAGES = 10


def get_pdf_hash(pdf_bytes: bytes) -> str:
    """MD5 hash of raw PDF bytes — used as a cache key."""
    return hashlib.md5(pdf_bytes).hexdigest()


def load_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Extract text from each page of a PDF.

    Returns:
        List of dicts: [{"page_num": int, "text": str}, ...]

    Raises:
        ValueError: if page count > MAX_PAGES or no text could be extracted.
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)

            if total_pages > MAX_PAGES:
                raise ValueError(
                    f"❌ PDF has **{total_pages} pages**. "
                    f"Maximum allowed is **{MAX_PAGES} pages**. "
                    "Please upload a shorter document."
                )

            pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append({"page_num": i + 1, "text": text})

    except ValueError:
        raise  # re-raise our own errors cleanly
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}") from e

    if not pages:
        raise ValueError(
            "❌ Could not extract any readable text from this PDF. "
            "The file may be scanned/image-based or corrupted."
        )

    return pages
