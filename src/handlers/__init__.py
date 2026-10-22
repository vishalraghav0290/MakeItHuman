"""
Document handlers package.
"""
from src.handlers.base import BaseDocumentHandler
from src.handlers.docx_handler import DocxHandler
from src.handlers.pdf_handler import PDFHandler
from src.handlers.pptx_handler import PPTXHandler

__all__ = ["BaseDocumentHandler", "DocxHandler", "PDFHandler", "PPTXHandler"]