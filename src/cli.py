"""Typer + Rich CLI entry point for doc-humanizer-cli (v4.0)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from src.detector import AcademicAIDetector, DetectionMetrics
from src.handlers.docx_handler import DocxHandler
from src.handlers.pdf_handler import PDFHandler
from src.humanizer import AcademicHumanizer, HumanizationResult

app = typer.Typer(
    name="doc-humanizer",
    help="Turnitin-Calibrated Academic AI Scorer & Dense Clause-Mutating Humanizer.",
    add_completion=False,
)
console = Console()
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def _validate_paths(input_path: Path, output_path: Path | None = None) -> tuple[str, Path]:
    if not input_path.exists() or not input_path.is_file():
        console.print(f"[bold red]Error:[/bold red] Input file not found: '{input_path}'")
        raise typer.Exit(code=1)

    in_ext = input_path.suffix.lower()
    if in_ext not in SUPPORTED_EXTENSIONS:
        console.print(
            f"[bold red]Error:[/bold red] Unsupported file format '{in_ext}'. "
            f"Only {', '.join(sorted(SUPPORTED_EXTENSIONS))} are supported."
        )
        raise typer.Exit(code=1)

    resolved_out = output_path or input_path.with_name(f"{input_path.stem}_humanized{in_ext}")
    if resolved_out.suffix.lower() != in_ext:
        console.print(
            f"[bold red]Strict Format Parity Violation:[/bold red] Input is '{in_ext}' "
            f"but output was specified as '{resolved_out.suffix}'. Output MUST be '{in_ext}'."
        )
        raise typer.Exit(code=1)

    return in_ext, resolved_out


def _format_risk_color(ai_prob: float, threshold: float = 15.0) -> str:
    if ai_prob <= threshold:
        return f"[bold green]{ai_prob:.1f}% (*)[/bold green]"
    if ai_prob < 25.0:
        return f"[bold yellow]{ai_prob:.1f}%[/bold yellow]"
    return f"[bold red]{ai_prob:.1f}%[/bold red]"


@app.command("scan")
def scan_command(
    input_path: Annotated[Path, typer.Argument(help="Path to .pdf or .docx file to scan.")],
    threshold: Annotated[float, typer.Option("--threshold", "-t")] = 15.0,
) -> None:
    """Scan a .pdf or .docx document and report Turnitin-calibrated AI probability."""
    ext, _ = _validate_paths(input_path)
    detector = AcademicAIDetector()
    handler = DocxHandler(detector) if ext == ".docx" else PDFHandler(detector)
    blocks = handler.extract_blocks(input_path)
    doc_metrics = detector.score_document(blocks)

    table = Table(title=f"Turnitin Diagnostic Report: {input_path.name}", header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Score / Value", justify="right")
    table.add_row("Turnitin Overall AI Score", _format_risk_color(doc_metrics.ai_probability, threshold))
    table.add_row("Cyan Channel (Generative AI)", f"[cyan]{doc_metrics.cyan_ai_score:.1f}%[/cyan]")
    table.add_row("Purple Channel (Paraphrased AI)", f"[magenta]{doc_metrics.purple_paraphrase_score:.1f}%[/magenta]")
    table.add_row("Burstiness (CV + Lag-1 Delta)", f"{doc_metrics.burstiness:.3f}")
    table.add_row("Perplexity Proxy (Surprisal)", f"{doc_metrics.perplexity_proxy:.2f}")
    table.add_row("Total Prose Words Analyzed", str(doc_metrics.word_count))
    console.print(table)


@app.command("humanize")
def humanize_command(
    input_path: Annotated[Path, typer.Argument(help="Input .pdf or .docx document path.")],
    output_path: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path (defaults to <name>_humanized.<ext>)."),
    ] = None,
    threshold: Annotated[float, typer.Option("--threshold", "-t")] = 15.0,
    max_loops: Annotated[int, typer.Option("--max-loops", "-l")] = 4,
    style: Annotated[str, typer.Option("--style", "-s")] = "academic",
) -> None:
    """Humanize all non-heading prose blocks in a .pdf or .docx document with 100% layout parity."""
    ext, resolved_output = _validate_paths(input_path, output_path)
    detector = AcademicAIDetector()
    humanizer = AcademicHumanizer(
        detector=detector,
        threshold=threshold,
        max_loops=max_loops,
        style=style,
    )

    handler = DocxHandler(detector, humanizer) if ext == ".docx" else PDFHandler(detector, humanizer)
    console.print(f"Processing: [bold cyan]{input_path.name}[/bold cyan] -> [bold green]{resolved_output.name}[/bold green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Mutating N-grams & clause structures...", total=1)

        def _update_bar(completed: int, total: int) -> None:
            progress.update(task_id, total=max(1, total), completed=completed)

        results: list[HumanizationResult] = handler.process_document(
            input_path=input_path,
            output_path=resolved_output,
            progress_callback=_update_bar,
        )

    before_doc = detector.score_document([r.original_text for r in results])
    after_doc = detector.score_document([r.humanized_text for r in results])
    modified_count = sum(1 for r in results if r.modified)

    table = Table(title="Before vs. After Telemetry", header_style="bold cyan")
    table.add_column("Metric", style="bold white")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_row("Overall AI Score", _format_risk_color(before_doc.ai_probability, threshold), _format_risk_color(after_doc.ai_probability, threshold))
    table.add_row("Burstiness", f"{before_doc.burstiness:.3f}", f"{after_doc.burstiness:.3f}")
    table.add_row("Perplexity Proxy", f"{before_doc.perplexity_proxy:.1f}", f"{after_doc.perplexity_proxy:.1f}")
    table.add_row("Active Rewrites", f"{len(results)} total blocks", f"[bold green]{modified_count} / {len(results)} rewritten[/bold green]")
    console.print(table)

    console.print(
        Panel(
            f"[bold green]Success! Humanized document saved to:[/bold green]\n{resolved_output.resolve()}\n\n"
            f"Active Rewrites: [bold cyan]{modified_count} / {len(results)}[/bold cyan] blocks/paragraphs modified.",
            border_style="green",
        )
    )


def main():
    app()


if __name__ == "__main__":
    app()
