from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

class BaseDocumentHandler(ABC):
    """
    Abstract base handler establishing the contract for document extractors
    and reconstructors across various file formats.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    @abstractmethod
    def extract_text(self) -> str:
        """
        Extracts raw textual content from the document.
        """
        pass

    @abstractmethod
    def process_and_save(self, output_path: Path, transform_func: Callable[[str], str]) -> None:
        """
        Applies the transformation function to the text in the document
        while preserving document formatting/structure where possible,
        and saves to the output path.
        """
        pass
