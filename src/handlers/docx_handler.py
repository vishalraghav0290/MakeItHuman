import logging
from pathlib import Path
from typing import Callable
from docx import Document
from src.handlers.base import BaseDocumentHandler

logger = logging.getLogger(__name__)

class DocxHandler(BaseDocumentHandler):
    """Handles extraction for Microsoft Word (.docx) files."""

    def extract_text(self) -> str:
        doc = Document(self.file_path)
        full_text = [p.text for p in doc.paragraphs if p.text]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text:
                        full_text.append(cell.text)
        return "\n".join(full_text)

    def process_and_save(self, output_path: Path, transform_func: Callable[[str], str]) -> None:
        pass
