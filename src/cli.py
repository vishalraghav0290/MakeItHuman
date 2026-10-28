import sys
from pathlib import Path
import click
from src.detector import AIDetector
from src.handlers.docx_handler import DocxHandler

@click.group()
def main():
    pass

@main.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
def scan(file: Path):
    handler = DocxHandler(file)
    text = handler.extract_text()
    detector = AIDetector()
    res = detector.analyze(text)
    print("AI Score:", res["score"])

if __name__ == "__main__":
    main()
