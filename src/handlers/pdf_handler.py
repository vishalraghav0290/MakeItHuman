"""
High-Speed Pixel-Perfect PDF Engine using PyMuPDF (fitz) (v3.0).
Uses a single persistent scratch page for sub-pixel bounding-box fitting and
parallelizes paragraph humanization across CPU threads.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from src.detector import AcademicAIDetector
from src.humanizer import AcademicHumanizer, HumanizationResult


@dataclass
class PDFTextBlock:
    page_index: int
    bbox: tuple[float, float, float, float]
    text: str
    font_code: str
    css_family: str
    font_size: float
    line_height_ratio: float
    is_bold: bool
    is_italic: bool
    rgb_color: tuple[float, float, float]
    align: int


class PDFHandler:
    """Fast parallel PDF handler with zero-destruction vector & image preservation."""

    def __init__(
        self,
        detector: AcademicAIDetector | None = None,
        humanizer: AcademicHumanizer | None = None,
        max_workers: int = 8,
    ) -> None:
        if isinstance(detector, (str, Path)):
            self.file_path = Path(detector)
            self.detector = AcademicAIDetector()
        else:
            self.detector = detector or AcademicAIDetector()
        self.humanizer = humanizer
        self.max_workers = max_workers

    @staticmethod
    def _srgb_int_to_floats(color_int: int) -> tuple[float, float, float]:
        r = ((color_int >> 16) & 255) / 255.0
        g = ((color_int >> 8) & 255) / 255.0
        b = (color_int & 255) / 255.0
        return (round(r, 4), round(g, 4), round(b, 4))

    @staticmethod
    def _map_font_attributes(raw_font: str, flags: int) -> tuple[str, str, bool, bool]:
        lowered = raw_font.lower()
        is_bold = bool(flags & 16) or any(k in lowered for k in ("bold", "medi", "cmbx", "heavy", "black"))
        is_italic = bool(flags & 2) or any(k in lowered for k in ("ital", "obli", "cmti", "sl"))

        is_mono = any(k in lowered for k in ("cour", "mono", "consol", "cmtt", "typewriter"))
        is_sans = any(k in lowered for k in ("helv", "arial", "sans", "calibri", "cmss", "roboto", "inter"))

        if is_mono:
            css = "monospace"
            code = "cobi" if (is_bold and is_italic) else ("cobo" if is_bold else ("coit" if is_italic else "cour"))
        elif is_sans:
            css = "sans-serif"
            code = "hebi" if (is_bold and is_italic) else ("hebo" if is_bold else ("heit" if is_italic else "helv"))
        else:
            css = "serif"
            code = "tibi" if (is_bold and is_italic) else ("tibo" if is_bold else ("tiit" if is_italic else "tiro"))

        return code, css, is_bold, is_italic

    def _detect_block_alignment(self, lines: list[dict], block_bbox: tuple[float, float, float, float]) -> int:
        if len(lines) < 3:
            return fitz.TEXT_ALIGN_LEFT
        block_width = block_bbox[2] - block_bbox[0]
        if block_width <= 80.0:
            return fitz.TEXT_ALIGN_LEFT

        non_terminal = lines[:-1]
        right_gaps = [abs(block_bbox[2] - float(ln["bbox"][2])) for ln in non_terminal]
        tight_right_lines = sum(1 for gap in right_gaps if gap <= 6.5)

        if tight_right_lines / len(non_terminal) >= 0.65:
            return fitz.TEXT_ALIGN_JUSTIFY
        return fitz.TEXT_ALIGN_LEFT

    def _extract_page_blocks(self, page: fitz.Page, page_index: int) -> list[PDFTextBlock]:
        page_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
        extracted: list[PDFTextBlock] = []

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            lines = block.get("lines", [])
            if not lines:
                continue

            line_strings: list[str] = []
            font_size_weights: dict[float, int] = {}
            font_sig_weights: dict[tuple[str, int], int] = {}
            color_weights: dict[int, int] = {}
            line_y_origins: list[float] = []

            for line in lines:
                l_bbox = line.get("bbox", (0.0, 0.0, 0.0, 0.0))
                line_y_origins.append(float(l_bbox[1]))
                span_texts: list[str] = []

                for span in line.get("spans", []):
                    txt = span.get("text", "")
                    if not txt:
                        continue
                    span_texts.append(txt)
                    w = len(txt.strip()) or 1
                    sz = round(float(span.get("size", 10.5)), 2)
                    fn = str(span.get("font", "Times-Roman"))
                    fl = int(span.get("flags", 0))
                    col = int(span.get("color", 0))

                    font_size_weights[sz] = font_size_weights.get(sz, 0) + w
                    font_sig_weights[(fn, fl)] = font_sig_weights.get((fn, fl), 0) + w
                    color_weights[col] = color_weights.get(col, 0) + w

                joined_line = "".join(span_texts).strip()
                if joined_line:
                    if line_strings and line_strings[-1].endswith("-") and not line_strings[-1].endswith("--"):
                        line_strings[-1] = line_strings[-1][:-1] + joined_line
                    else:
                        line_strings.append(joined_line)

            full_text = " ".join(line_strings).strip()
            if not full_text:
                continue

            dominant_size = max(font_size_weights, key=font_size_weights.get) if font_size_weights else 10.5
            dominant_fn, dominant_flags = (
                max(font_sig_weights, key=font_sig_weights.get)
                if font_sig_weights
                else ("Times-Roman", 0)
            )
            dominant_color = max(color_weights, key=color_weights.get) if color_weights else 0

            if len(line_y_origins) >= 2 and dominant_size > 0:
                diffs = [
                    line_y_origins[i] - line_y_origins[i - 1]
                    for i in range(1, len(line_y_origins))
                    if (line_y_origins[i] - line_y_origins[i - 1]) > dominant_size * 0.7
                ]
                avg_leading = sum(diffs) / len(diffs) if diffs else dominant_size * 1.2
                lh_ratio = round(max(1.05, min(1.65, avg_leading / dominant_size)), 3)
            else:
                lh_ratio = 1.18

            bbox_raw = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
            bbox = (float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3]))

            font_code, css_family, is_bold, is_italic = self._map_font_attributes(dominant_fn, dominant_flags)
            align = self._detect_block_alignment(lines, bbox)

            extracted.append(
                PDFTextBlock(
                    page_index=page_index,
                    bbox=bbox,
                    text=full_text,
                    font_code=font_code,
                    css_family=css_family,
                    font_size=dominant_size,
                    line_height_ratio=lh_ratio,
                    is_bold=is_bold,
                    is_italic=is_italic,
                    rgb_color=self._srgb_int_to_floats(dominant_color),
                    align=align,
                )
            )

        return extracted

    def extract_blocks(self, input_path: Path) -> list[str]:
        doc = fitz.open(str(input_path))
        try:
            texts: list[str] = []
            for page_idx in range(len(doc)):
                blocks = self._extract_page_blocks(doc[page_idx], page_idx)
                texts.extend(b.text for b in blocks if b.text.strip())
            return texts
        finally:
            doc.close()

    def extract_text(self) -> str:
        """Backward compatibility method for legacy callers."""
        if hasattr(self, "file_path"):
            return "\n".join(self.extract_blocks(self.file_path))
        raise AttributeError("No file_path provided at initialization")

    def _insert_fitted_text(
        self,
        page: fitz.Page,
        scratch_doc: fitz.Document,
        block: PDFTextBlock,
        new_text: str,
    ) -> None:
        rect = fitz.Rect(block.bbox)
        target_rect = fitz.Rect(rect.x0, rect.y0, rect.x1 + 1.2, rect.y1 + 1.8)

        deltas = [0.0, -0.10, 0.10, -0.20, -0.35, -0.50, -0.75, -1.00, -1.25]
        candidate_sizes = [round(block.font_size + d, 2) for d in deltas if block.font_size + d >= 5.5]

        best_size = candidate_sizes[-1]
        scratch_page = scratch_doc[0]

        for sz in candidate_sizes:
            shape = scratch_page.new_shape()
            rc = shape.insert_textbox(
                target_rect,
                new_text,
                fontsize=sz,
                fontname=block.font_code,
                color=block.rgb_color,
                align=block.align,
            )
            if rc >= 0:
                best_size = sz
                break

        page.insert_textbox(
            target_rect,
            new_text,
            fontsize=best_size,
            fontname=block.font_code,
            color=block.rgb_color,
            align=block.align,
        )

    def process_document(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[HumanizationResult]:
        if self.humanizer is None:
            self.humanizer = AcademicHumanizer(detector=self.detector)

        doc = fitz.open(str(input_path))
        scratch_doc = fitz.open()
        scratch_doc.new_page(width=1200, height=1600)

        try:
            all_page_blocks: list[tuple[int, PDFTextBlock]] = []
            for page_idx in range(len(doc)):
                for b in self._extract_page_blocks(doc[page_idx], page_idx):
                    all_page_blocks.append((page_idx, b))

            total = len(all_page_blocks)
            results: list[HumanizationResult | None] = [None] * total
            completed_count = 0

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future_to_idx = {
                    pool.submit(self.humanizer.humanize_paragraph, b.text): i
                    for i, (_, b) in enumerate(all_page_blocks)
                }
                for future in future_to_idx:
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, total)

            redactions_by_page: dict[int, list[tuple[PDFTextBlock, str]]] = {}
            final_results: list[HumanizationResult] = []

            for (page_idx, block), res in zip(all_page_blocks, results):
                assert res is not None
                final_results.append(res)
                if res.modified:
                    redactions_by_page.setdefault(page_idx, []).append(
                        (block, res.humanized_text)
                    )

            for page_idx, modifications in redactions_by_page.items():
                page = doc[page_idx]
                for block, _ in modifications:
                    r = fitz.Rect(block.bbox)
                    safe_redact_rect = fitz.Rect(r.x0, r.y0 + 0.3, r.x1, r.y1 - 0.3)
                    page.add_redact_annot(safe_redact_rect, fill=False)

                page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_NONE,
                    graphics=getattr(fitz, "PDF_REDACT_LINE_ART_NONE", 0),
                )

                for block, new_text in modifications:
                    self._insert_fitted_text(page, scratch_doc, block, new_text)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(output_path), garbage=3, deflate=True)
            return final_results
        finally:
            scratch_doc.close()
            doc.close()

    def process_and_save(self, output_path: Path, transform_func: Callable[[str], str]) -> None:
        """Backward compatibility method for legacy callers."""
        if not hasattr(self, "file_path"):
            raise AttributeError("No file_path provided at initialization")
        if self.humanizer is None:
            self.humanizer = AcademicHumanizer(detector=self.detector)
        self.process_document(self.file_path, output_path)
