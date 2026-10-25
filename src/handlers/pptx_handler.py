import logging
from pathlib import Path
from typing import Callable
from pptx import Presentation
from src.handlers.base import BaseDocumentHandler

logger = logging.getLogger(__name__)

class PPTXHandler(BaseDocumentHandler):
    """Handles extraction for PowerPoint presentations."""

    def extract_text(self) -> str:
        prs = Presentation(self.file_path)
        extracted = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for p in shape.text_frame.paragraphs:
                        if p.text:
                            extracted.append(p.text)
        return "\n".join(extracted)

    def process_and_save(self, output_path: Path, transform_func: Callable[[str], str]) -> None:
        pass
