import logging
from pathlib import Path
from typing import Callable
from docx import Document
from src.handlers.base import BaseDocumentHandler

logger = logging.getLogger(__name__)

class DocxHandler(BaseDocumentHandler):
    """
    Handles extraction and in-place textual replacement for Microsoft Word (.docx) files.
    """

    def extract_text(self) -> str:
        try:
            doc = Document(self.file_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text:
                    full_text.append(para.text)
            
            # Also extract from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text:
                            full_text.append(cell.text)

            return "\n".join(full_text)
        except Exception as e:
            logger.error(f"Failed to extract text from DOCX: {e}")
            raise

    def process_and_save(self, output_path: Path, transform_func: Callable[[str], str]) -> None:
        try:
            doc = Document(self.file_path)

            # Process body paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    para.text = transform_func(para.text)

            # Process table cells
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            if para.text.strip():
                                para.text = transform_func(para.text)

            doc.save(str(output_path))
            logger.info(f"Successfully processed and saved DOCX to {output_path}")
        except Exception as e:
            logger.error(f"Error processing DOCX: {e}")
            raise