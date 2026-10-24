import logging
from pathlib import Path
from typing import Callable
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.handlers.base import BaseDocumentHandler

logger = logging.getLogger(__name__)

class PDFHandler(BaseDocumentHandler):
    """
    Handles PDF text extraction and simplified generation for transformed content.
    Note: Full layout-preserving PDF replacement is inherently lossy without specialized OCR/editing engines.
    """

    def extract_text(self) -> str:
        try:
            reader = PdfReader(self.file_path)
            extracted_pages = []
            for idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    extracted_pages.append(text)
            return "\n".join(extracted_pages)
        except Exception as e:
            logger.error(f"Failed to read PDF: {e}")
            raise

    def process_and_save(self, output_path: Path, transform_func: Callable[[str], str]) -> None:
        try:
            # Extract and transform text
            original_text = self.extract_text()
            transformed_text = transform_func(original_text)

            # Simple rewrite strategy using ReportLab canvas
            c = canvas.Canvas(str(output_path), pagesize=letter)
            width, height = letter
            y_cursor = height - 50
            margin = 50
            line_height = 14

            c.setFont("Helvetica", 10)

            for line in transformed_text.splitlines():
                if y_cursor <= margin:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y_cursor = height - 50
                
                # Basic wrapping check (simplification for raw canvas)
                if len(line) > 90:
                    chunks = [line[i:i+90] for i in range(0, len(line), 90)]
                    for chunk in chunks:
                        c.drawString(margin, y_cursor, chunk)
                        y_cursor -= line_height
                else:
                    c.drawString(margin, y_cursor, line)
                    y_cursor -= line_height

            c.save()
            logger.info(f"Generated rewritten PDF at {output_path}")
        except Exception as e:
            logger.error(f"Error writing output PDF: {e}")
            raise