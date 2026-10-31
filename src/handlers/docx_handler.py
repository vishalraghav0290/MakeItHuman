"""
Deep-XML Word (.docx) Engine (v4.0).
Iterates all <w:p> nodes across the entire XML tree (including <w:sdt> content controls,
tables, lists, and textboxes) while preserving 100% of Run-level <w:rPr> formatting.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from src.detector import AcademicAIDetector
from src.humanizer import AcademicHumanizer, HumanizationResult


class DocxHandler:
    """Extracts and rewrites every <w:p> element in the document body without losing formatting."""

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

    def collect_all_paragraphs(self, doc: Document) -> list[Paragraph]:
        all_paragraphs: list[Paragraph] = []
        seen_p: set[int] = set()

        # Walk every <w:p> in body (covers normal paragraphs, tables, SDT content controls, and textboxes)
        for p_elem in doc.element.body.iter(qn("w:p")):
            p_id = id(p_elem)
            if p_id in seen_p:
                continue
            p_obj = Paragraph(p_elem, doc)
            if p_obj.text and p_obj.text.strip():
                seen_p.add(p_id)
                all_paragraphs.append(p_obj)

        return all_paragraphs

    @classmethod
    def redistribute_text_across_runs(cls, paragraph: Paragraph, new_text: str) -> None:
        t_nodes = [
            elem for elem in paragraph._p.iter(qn("w:t"))
            if elem.text and len(elem.text) > 0
        ]

        if not t_nodes:
            if paragraph.runs:
                paragraph.runs[0].text = new_text
            else:
                paragraph.add_run(new_text)
            return

        if len(t_nodes) == 1:
            t_nodes[0].text = new_text
            return

        orig_lengths = [len(t.text) for t in t_nodes]
        total_orig_chars = max(1, sum(orig_lengths))
        words = new_text.split(" ")
        total_words = len(words)

        word_cursor = 0
        cumulative_chars = 0

        for pos, t_node in enumerate(t_nodes):
            is_last = pos == len(t_nodes) - 1
            cumulative_chars += orig_lengths[pos]

            if is_last:
                chunk_words = words[word_cursor:]
            else:
                target_end = round((cumulative_chars / total_orig_chars) * total_words)
                target_end = max(word_cursor, min(total_words, target_end))
                chunk_words = words[word_cursor:target_end]
                word_cursor = target_end

            segment = " ".join(chunk_words)
            if segment and not is_last:
                segment += " "
                t_node.set(qn("xml:space"), "preserve")
            t_node.text = segment

    def extract_blocks(self, input_path: Path) -> list[str]:
        doc = Document(str(input_path))
        paragraphs = self.collect_all_paragraphs(doc)
        return [p.text.strip() for p in paragraphs if p.text.strip()]

    def extract_text(self) -> str:
        """Backward compatibility method for legacy callers."""
        if hasattr(self, "file_path"):
            return "\n".join(self.extract_blocks(self.file_path))
        raise AttributeError("No file_path provided at initialization")

    def process_and_save(self, output_path: Path, transform_func: Callable[[str], str]) -> None:
        """Backward compatibility method for legacy callers."""
        if not hasattr(self, "file_path"):
            raise AttributeError("No file_path provided at initialization")
        doc = Document(str(self.file_path))
        paragraphs = self.collect_all_paragraphs(doc)
        for p in paragraphs:
            if p.text.strip():
                new_text = transform_func(p.text)
                self.redistribute_text_across_runs(p, new_text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))

    def process_document(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[HumanizationResult]:
        if self.humanizer is None:
            self.humanizer = AcademicHumanizer(detector=self.detector)

        doc = Document(str(input_path))
        paragraphs = self.collect_all_paragraphs(doc)
        total = len(paragraphs)
        raw_texts = [p.text for p in paragraphs]

        results: list[HumanizationResult | None] = [None] * total
        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_idx = {
                pool.submit(self.humanizer.humanize_paragraph, txt): i
                for i, txt in enumerate(raw_texts)
            }
            for future in future_to_idx:
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total)

        final_results: list[HumanizationResult] = []
        for paragraph, res in zip(paragraphs, results):
            assert res is not None
            if res.modified:
                self.redistribute_text_across_runs(paragraph, res.humanized_text)
            final_results.append(res)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        return final_results