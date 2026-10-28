import sys
import logging
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.detector import AIDetector
from src.humanizer import TextHumanizer
from src.handlers.docx_handler import DocxHandler
from src.handlers.pdf_handler import PDFHandler
from src.handlers.pptx_handler import PPTXHandler

# Setup console for rich outputs
console = Console()

# Configure basic logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("doc-humanizer")

def get_handler(file_path: Path):
    """
    Factory resolving the appropriate document handler by file extension.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".docx":
        return DocxHandler(file_path)
    elif suffix == ".pdf":
        return PDFHandler(file_path)
    elif suffix == ".pptx":
        return PPTXHandler(file_path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported: .pdf, .docx, .pptx")

@click.group()
@click.option("--verbose", is_flag=True, help="Enable debug logging output.")
def main(verbose: bool):
    """Doc Humanizer CLI - Detect and humanize AI artifacts in documents."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

@main.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
def scan(file: Path):
    """Scan a document and display an AI-likelihood score and markers."""
    try:
        handler = get_handler(file)
        console.print(f"[cyan]Reading content from:[/] {file.name}")
        text = handler.extract_text()

        if not text.strip():
            console.print("[yellow]Warning:[/] No extractable text found in file.")
            return

        detector = AIDetector()
        results = detector.analyze(text)

        # Output formatting
        score = results["score"]
        score_color = "green" if score < 0.3 else "yellow" if score < 0.7 else "red"
        
        table = Table(title="Document AI Analysis", show_header=True)
        table.add_column("Metric", style="dim")
        table.add_column("Value")

        table.add_row("Likelihood Score", f"[{score_color}]{score * 100:.1f}%[/{score_color}]")
        table.add_row("Sentence Variance", str(results["sentence_variance"]))
        table.add_row("Detected Buzzwords", ", ".join(results["matches"]) if results["matches"] else "None")

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error during scan:[/] {e}")
        sys.exit(1)

@main.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Output path for the modified file.")
def humanize(input_file: Path, output: Path):
    """Rewrite robotic phrasing and restructure document content."""
    try:
        handler = get_handler(input_file)
        
        # Default output name if not provided
        if not output:
            output = input_file.parent / f"{input_file.stem}_humanized{input_file.suffix}"

        console.print(f"[cyan]Processing:[/] {input_file.name} -> {output.name}")
        
        humanizer = TextHumanizer()
        handler.process_and_save(output, transform_func=humanizer.humanize)

        console.print(Panel(f"[bold green]Success![/] Humanized document saved to:\n{output.resolve()}", expand=False))

    except Exception as e:
        console.print(f"[bold red]Error during processing:[/] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()