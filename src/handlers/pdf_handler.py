import logging
from pathlib import Path
from typing import Callable
from pypdf import PdfReader
from src.handlers.base import BaseDocumentHandler

logger = logging.getLogger(__name__)

class PDFHandler(BaseDocumentHandler):
    """Handles PDF text extraction."""

    def extract_text(self) -> str:
        reader = PdfReader(self.file_path)
        pages = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(pages)

    def process_and_save(self, output_path: Path, transform_func: Callable[[str], str]) -> None:
        pass
