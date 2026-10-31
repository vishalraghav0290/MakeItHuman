# Doc Humanizer CLI

A CLI tool to detect and humanize AI-generated text in documents (`.pdf`, `.docx`, `.pptx`).

## Features

- **Heuristic-based AI Detection**: Identifies overused LLM transition words, buzzwords, and uniform burstiness/sentence variance.
- **Document Handlers**: Modular extractors and processors for Microsoft Word (`.docx`), PowerPoint (`.pptx`), and PDF (`.pdf`).
- **Humanizer Engine**: Replaces robotic buzzwords and repetitive qualifiers with natural conversational phrasing.
- **Rich CLI Interface**: Formatted terminal output with tables, metrics, and color-coded likelihood scores.

## Quick Setup

Activate the existing virtual environment:
```bash
source .venv/bin/activate
```

*(Or if setting up from scratch with uv)*:
```bash
uv venv
uv pip install -e .
source .venv/bin/activate
```


## Usage

### 1. Scan a document for AI likelihood
```bash
doc-humanizer scan report.docx
doc-humanizer scan paper.pdf
doc-humanizer scan presentation.pptx
```

### 2. Humanize a document
```bash
# Auto-generates an output with '_humanized' suffix
doc-humanizer humanize report.docx

# Or specify a custom output path
doc-humanizer humanize presentation.pptx -o humanized_deck.pptx
```

