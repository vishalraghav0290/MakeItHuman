import logging
from pathlib import Path
from typing import Callable
from pptx import Presentation
from src.handlers.base import BaseDocumentHandler

logger = logging.getLogger(__name__)

class PPTXHandler(BaseDocumentHandler):
    """
    Handles extraction and text-frame level transformations for PowerPoint presentations.
    """

    def extract_text(self) -> str:
        try:
            prs = Presentation(self.file_path)
            extracted_text = []

            for slide in prs.slides:
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            if paragraph.text:
                                extracted_text.append(paragraph.text)

            return "\n".join(extracted_text)
        except Exception as e:
            logger.error(f"Failed to read PPTX: {e}")
            raise

    def process_and_save(self, output_path: Path, transform_func: Callable[[str], str]) -> None:
        try:
            prs = Presentation(self.file_path)

            for slide in prs.slides:
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            if paragraph.text.strip():
                                paragraph.text = transform_func(paragraph.text)

            prs.save(str(output_path))
            logger.info(f"Successfully processed and saved PPTX to {output_path}")
        except Exception as e:
            logger.error(f"Error processing PPTX: {e}")
            raise