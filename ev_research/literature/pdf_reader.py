"""Read text from PDF files page by page.

Only the text layer is read.  Scanned PDFs (images without a text layer) are
detected and reported; OCR is not performed.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field


class PDFReadError(Exception):
    """Raised when a PDF cannot be read or contains no extractable text."""


@dataclass
class PDFDocument:
    filename: str
    pages: list[str]                      # pages[i] = text of page i+1
    metadata: dict = field(default_factory=dict)
    title_candidate: str = ""             # largest-font text on page 1
    title_candidate_page: int | None = None
    lines_after_title: list[str] = field(default_factory=list)

    @property
    def n_pages(self) -> int:
        return len(self.pages)


def _clean_text(text: str) -> str:
    text = text.replace("­", "")              # soft hyphen
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _largest_font_lines(page) -> tuple[str, list[str]]:
    """Return text set in the largest font on the page (title heuristic) and
    the next few lines that follow it (author heuristic)."""
    try:
        words = page.extract_words(extra_attrs=["size"], keep_blank_chars=False)
    except Exception:  # pragma: no cover - pdfplumber internals
        return "", []
    words = [w for w in words if w.get("text", "").strip()]
    if not words:
        return "", []
    max_size = max(round(w["size"], 1) for w in words)
    title_words = [w for w in words if round(w["size"], 1) >= max_size - 0.5]
    # Only accept a title if the largest font is clearly larger than body text
    sizes = sorted(round(w["size"], 1) for w in words)
    median = sizes[len(sizes) // 2]
    if max_size < median * 1.15 or len(title_words) > 40:
        return "", []
    title = " ".join(w["text"] for w in sorted(title_words, key=lambda w: (round(w["top"]), w["x0"])))
    bottom = max(w["bottom"] for w in title_words)
    # Group following words into lines
    rest = sorted([w for w in words if w["top"] > bottom], key=lambda w: (round(w["top"]), w["x0"]))
    lines: dict[int, list[str]] = {}
    for w in rest:
        lines.setdefault(round(w["top"]), []).append(w["text"])
    following = [" ".join(v) for _, v in sorted(lines.items())][:3]
    return _clean_text(title), following


def read_pdf(data: bytes, filename: str = "uploaded.pdf") -> PDFDocument:
    """Read a PDF from bytes. Raises PDFReadError with a readable message."""
    if not data:
        raise PDFReadError(f"'{filename}' is empty.")
    if not data[:1024].lstrip().startswith(b"%PDF"):
        raise PDFReadError(f"'{filename}' does not look like a PDF file.")
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise PDFReadError("The 'pdfplumber' package is not installed.") from exc

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [_clean_text(p.extract_text() or "") for p in pdf.pages]
            metadata = {k: str(v) for k, v in (pdf.metadata or {}).items()}
            title, following = ("", [])
            if pdf.pages:
                title, following = _largest_font_lines(pdf.pages[0])
    except Exception as exc:
        raise PDFReadError(
            f"Could not read '{filename}'. The file may be damaged or password-protected. ({exc})"
        ) from exc

    if not pages:
        raise PDFReadError(f"'{filename}' has no pages.")
    total_chars = sum(len(p) for p in pages)
    if total_chars < 200:
        raise PDFReadError(
            f"'{filename}' contains almost no extractable text. It is probably a scanned "
            "image. Run OCR on it first (for example with Adobe Acrobat or 'ocrmypdf') "
            "and upload the OCR'd version, or enter the details manually."
        )
    return PDFDocument(
        filename=filename,
        pages=pages,
        metadata=metadata,
        title_candidate=title,
        title_candidate_page=1 if title else None,
        lines_after_title=following,
    )
