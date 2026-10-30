"""
Turnitin-Calibrated 6-Axis Statistical & Structural AI Detector (v3.0).
Zero runtime regex compilation overhead; pre-compiled pattern tables for sub-millisecond scoring.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field


LLM_LEXICAL_WEIGHTS: dict[str, float] = {
    "delve into": 4.5,
    "delves into": 4.5,
    "testament to": 4.2,
    "plays a crucial role": 3.8,
    "crucial role": 3.2,
    "vital role": 3.0,
    "pivotal role": 3.5,
    "moreover": 2.4,
    "furthermore": 2.4,
    "consequently": 1.8,
    "it is important to note": 4.0,
    "it is worth noting": 3.8,
    "tapestry": 4.8,
    "intricate interplay": 4.6,
    "shed light on": 3.4,
    "sheds light on": 3.4,
    "pave the way": 3.5,
    "underscores the importance": 4.1,
    "underscores": 2.6,
    "in conclusion": 2.5,
    "a myriad of": 3.9,
    "paradigm shift": 3.4,
    "seamless integration": 3.7,
    "foster a sense of": 3.8,
    "holistic approach": 3.5,
    "groundbreaking": 3.1,
    "in the realm of": 3.9,
    "not only": 1.6,
    "meticulously": 3.6,
    "beacon of": 4.2,
    "ever-evolving": 4.0,
    "multifaceted": 3.3,
    "nuanced": 2.5,
    "comprehensive overview": 3.2,
    "notably": 1.7,
    "significantly": 1.5,
    "demonstrates that": 1.4,
    "exhibits": 1.3,
    "facilitates": 1.9,
    "utilizing": 2.1,
    "leveraging": 2.8,
    "synergy": 3.4,
    "cornerstone": 3.3,
    "paramount": 3.0,
    "poised to": 3.5,
    "navigating the": 3.4,
    "landscape of": 2.9,
    "this study": 1.5,
    "the results indicate": 1.8,
    "it is evident that": 2.6,
    "plays an important role": 2.7,
    "in order to": 1.4,
}

# Pre-compile all lexical patterns once at module import (eliminates 20,000+ runtime compiles)
COMPILED_LEXICAL_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    (phrase, re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE), weight)
    for phrase, weight in LLM_LEXICAL_WEIGHTS.items()
]

PREDICTABLE_COLLOCATIONS: set[tuple[str, str]] = {
    ("this", "study"), ("the", "results"), ("results", "indicate"), ("indicate", "that"),
    ("it", "is"), ("can", "be"), ("in", "this"), ("of", "the"), ("to", "the"),
    ("on", "the"), ("for", "the"), ("with", "the"), ("from", "the"), ("by", "the"),
    ("as", "a"), ("such", "as"), ("in", "order"), ("order", "to"), ("due", "to"),
    ("based", "on"), ("play", "a"), ("plays", "a"), ("has", "been"), ("have", "been"),
    ("was", "found"), ("were", "found"), ("is", "essential"), ("are", "essential"),
    ("crucial", "for"), ("important", "to"), ("leads", "to"), ("contributes", "to"),
    ("associated", "with"), ("compared", "to"), ("in", "addition"), ("as", "well"),
    ("well", "as"), ("a", "significant"), ("significant", "impact"), ("further", "research"),
}

SPINNING_ARTIFACTS: set[str] = {
    "markedly", "pronouncedly", "substantively", "aforementioned", "heretofore",
    "thereby", "whereby", "wherein", "vis-a-vis", "henceforth", "inasmuch",
}

ABBREV_PROTECT_REGEX = re.compile(
    r"\b(et\s+al|e\.g|i\.e|Fig|Figs|Eq|Eqs|Dr|Prof|Mr|Mrs|Ms|vs|etc|No|Vol|pp|ch|sec)\.",
    re.IGNORECASE,
)
SENTENCE_BOUNDARY_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"\'])")
WORD_REGEX = re.compile(r"\b[a-zA-Z0-9'-]+\b")
PUNCT_VARIETY_REGEX = re.compile(r"[;:—–\(\)\[\]\?]")
EMPIRICAL_MARKER_REGEX = re.compile(r"\b\d+(?:\.\d+)?%?|\bp\s*[<>=]\s*0\.\d+")


@dataclass
class DetectionMetrics:
    text: str
    ai_probability: float
    cyan_ai_score: float
    purple_paraphrase_score: float
    perplexity_proxy: float
    burstiness: float
    syntactic_symmetry: float
    synthetic_density: float
    flagged_phrases: list[str] = field(default_factory=list)
    sentence_count: int = 0
    word_count: int = 0


class AcademicAIDetector:
    """Fast, pre-compiled 6-axis Turnitin-calibrated AI detector."""

    def split_sentences(self, text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text.strip())
        if not cleaned:
            return []
        protected = ABBREV_PROTECT_REGEX.sub(r"\1<DOT>", cleaned)
        raw_parts = SENTENCE_BOUNDARY_REGEX.split(protected)
        return [s.replace("<DOT>", ".").strip() for s in raw_parts if len(s.strip()) > 1]

    def tokenize(self, text: str) -> list[str]:
        return [m.group(0).lower() for m in WORD_REGEX.finditer(text)]

    def _classify_opener_signature(self, sentence: str) -> str:
        words = self.tokenize(sentence)[:3]
        if not words:
            return "EMPTY"
        w0 = words[0]
        if w0 in {
            "however", "moreover", "furthermore", "consequently", "therefore", "additionally",
            "thus", "hence", "notably", "specifically", "similarly", "conversely", "meanwhile",
        }:
            return "CONJ_ADV"
        if w0 in {"the", "a", "an", "this", "these", "that", "those", "each", "every", "all"}:
            return "DET_NP"
        if w0 in {"although", "while", "whereas", "because", "since", "if", "when", "despite"}:
            return "SUB_CLAUSE"
        if w0 in {"in", "on", "at", "by", "for", "with", "from", "under", "through", "during", "across"}:
            return "PREP_PHRASE"
        if w0 in {"it", "we", "they", "he", "she", "one", "there", "our"}:
            return "PRON_SUBJ"
        if w0.endswith("ing") or w0.endswith("ed"):
            return "PARTICIPLE"
        return "LEXICAL_HEAD"

    def calculate_perplexity_proxy(self, tokens: list[str], text: str) -> float:
        if len(tokens) < 10:
            return 72.0

        bigrams = list(zip(tokens[:-1], tokens[1:]))
        predictable_hits = sum(1 for bg in bigrams if bg in PREDICTABLE_COLLOCATIONS)
        predictable_ratio = predictable_hits / max(1, len(bigrams))

        counts = Counter(tokens)
        hapax_ratio = sum(1 for c in counts.values() if c == 1) / len(tokens)

        word_lengths = [len(t) for t in tokens]
        len_std = statistics.stdev(word_lengths) if len(word_lengths) > 2 else 2.0

        semi_colon_dash = len(PUNCT_VARIETY_REGEX.findall(text))
        word_scale = max(1.0, len(tokens) / 100.0)
        punct_variety_bonus = min(18.0, (semi_colon_dash / word_scale) * 3.8)

        empirical_markers = len(EMPIRICAL_MARKER_REGEX.findall(text))
        empirical_bonus = min(14.0, (empirical_markers / word_scale) * 2.5)

        surprisal = (
            (hapax_ratio * 62.0)
            + (len_std * 7.5)
            + punct_variety_bonus
            + empirical_bonus
            - (predictable_ratio * 145.0)
        )
        return round(max(5.0, min(98.0, surprisal)), 2)

    def calculate_burstiness_and_pacing(self, sentences: list[str]) -> tuple[float, float]:
        if len(sentences) < 2:
            return 0.26, 0.75

        lengths = [len(self.tokenize(s)) for s in sentences]
        lengths = [L for L in lengths if L > 0]
        if len(lengths) < 2:
            return 0.26, 0.75

        mean_len = statistics.mean(lengths)
        stdev_len = statistics.stdev(lengths)
        cv = stdev_len / max(1.0, mean_len)

        lag_diffs = [abs(lengths[i] - lengths[i - 1]) for i in range(1, len(lengths))]
        mean_lag_diff = statistics.mean(lag_diffs) / max(1.0, mean_len)

        in_ai_band = sum(1 for L in lengths if 14 <= L <= 28)
        band_concentration = in_ai_band / len(lengths)

        composite_burstiness = (0.55 * cv) + (0.45 * mean_lag_diff)
        return round(composite_burstiness, 4), round(band_concentration, 4)

    def calculate_syntactic_symmetry(self, sentences: list[str]) -> float:
        if len(sentences) < 3:
            return 0.38

        openers = [self._classify_opener_signature(s) for s in sentences]
        opener_counts = Counter(openers)
        most_common_ratio = opener_counts.most_common(1)[0][1] / len(sentences)

        adjacent_repeats = sum(
            1 for i in range(1, len(openers)) if openers[i] == openers[i - 1]
        ) / (len(openers) - 1)

        comma_counts = [s.count(",") for s in sentences]
        comma_mode_ratio = Counter(comma_counts).most_common(1)[0][1] / len(sentences)

        symmetry = (0.45 * most_common_ratio) + (0.30 * adjacent_repeats) + (0.25 * comma_mode_ratio)
        return round(min(1.0, max(0.0, symmetry)), 4)

    def calculate_synthetic_and_purple_risk(
        self, text: str, tokens: list[str]
    ) -> tuple[float, float, list[str]]:
        if not tokens:
            return 0.0, 0.0, []

        word_count = len(tokens)
        flagged: list[str] = []
        weighted_sum = 0.0

        for phrase, pattern, weight in COMPILED_LEXICAL_PATTERNS:
            hits = len(pattern.findall(text))
            if hits > 0:
                flagged.append(phrase)
                weighted_sum += hits * weight

        density_score = (weighted_sum / max(1.0, word_count)) * 100.0

        spin_hits = sum(1 for t in tokens if t in SPINNING_ARTIFACTS)
        adv_ly_hits = sum(1 for t in tokens if len(t) > 6 and t.endswith("ly"))
        purple_raw = (
            (spin_hits * 3.5 + max(0.0, adv_ly_hits - (word_count * 0.035)) * 1.8) / word_count
        ) * 100.0

        return round(density_score, 3), round(min(100.0, purple_raw * 6.5), 2), flagged

    def score_text(self, text: str) -> DetectionMetrics:
        sentences = self.split_sentences(text)
        tokens = self.tokenize(text)
        word_count = len(tokens)

        if word_count < 12:
            return DetectionMetrics(
                text=text,
                ai_probability=0.0,
                cyan_ai_score=0.0,
                purple_paraphrase_score=0.0,
                perplexity_proxy=85.0,
                burstiness=0.58,
                syntactic_symmetry=0.20,
                synthetic_density=0.0,
                flagged_phrases=[],
                sentence_count=len(sentences),
                word_count=word_count,
            )

        perplexity_proxy = self.calculate_perplexity_proxy(tokens, text)
        burstiness, band_concentration = self.calculate_burstiness_and_pacing(sentences)
        symmetry = self.calculate_syntactic_symmetry(sentences)
        synth_density, purple_base, flagged = self.calculate_synthetic_and_purple_risk(text, tokens)

        burst_deficit = max(0.0, 0.54 - burstiness) * 6.8
        band_penalty = max(0.0, band_concentration - 0.40) * 3.8
        perp_deficit = max(0.0, (63.0 - perplexity_proxy) / 13.5)
        sym_penalty = max(0.0, (symmetry - 0.32) * 4.6)
        lex_penalty = min(4.5, synth_density * 0.30 + len(flagged) * 0.60)

        logit_z = (
            -2.05
            + burst_deficit
            + band_penalty
            + perp_deficit
            + sym_penalty
            + lex_penalty
        )
        cyan_prob = 100.0 / (1.0 + math.exp(-logit_z))

        purple_structural = (
            (band_concentration * 42.0)
            + (symmetry * 38.0)
            + (purple_base * 0.45)
            - (max(0.0, burstiness - 0.48) * 65.0)
        )
        purple_score = max(0.0, min(99.0, purple_structural))

        combined_prob = max(cyan_prob, cyan_prob * 0.65 + purple_score * 0.45)
        combined_prob = round(max(1.0, min(99.4, combined_prob)), 1)

        return DetectionMetrics(
            text=text,
            ai_probability=combined_prob,
            cyan_ai_score=round(cyan_prob, 1),
            purple_paraphrase_score=round(purple_score, 1),
            perplexity_proxy=perplexity_proxy,
            burstiness=round(burstiness, 3),
            syntactic_symmetry=round(symmetry, 3),
            synthetic_density=round(synth_density, 2),
            flagged_phrases=flagged,
            sentence_count=len(sentences),
            word_count=word_count,
        )

    def score_document(self, blocks: list[str]) -> DetectionMetrics:
        valid_blocks = [b.strip() for b in blocks if len(self.tokenize(b)) >= 12]
        if not valid_blocks:
            return self.score_text(" ".join(blocks))

        windows: list[str] = []
        current_group: list[str] = []
        current_words = 0

        for block in valid_blocks:
            w_len = len(self.tokenize(block))
            current_group.append(block)
            current_words += w_len
            if current_words >= 220:
                windows.append(" ".join(current_group))
                current_group = [current_group[-1]] if len(current_group) > 1 else []
                current_words = len(self.tokenize(current_group[0])) if current_group else 0

        if current_group and (not windows or current_words >= 30):
            windows.append(" ".join(current_group))
        if not windows:
            windows = [" ".join(valid_blocks)]

        window_metrics = [self.score_text(w) for w in windows]
        block_metrics = [self.score_text(b) for b in valid_blocks]

        total_words = max(1, sum(m.word_count for m in block_metrics))
        total_win_words = max(1, sum(m.word_count for m in window_metrics))

        accum = 0.0
        for m in window_metrics:
            if m.ai_probability >= 50.0:
                effective = min(100.0, m.ai_probability * 1.15)
            elif m.ai_probability >= 22.0:
                effective = m.ai_probability
            else:
                effective = m.ai_probability * 0.45
            accum += effective * m.word_count

        overall_ai = round(min(99.5, accum / total_win_words), 1)
        all_flagged: list[str] = []
        for m in block_metrics:
            for p in m.flagged_phrases:
                if p not in all_flagged:
                    all_flagged.append(p)

        return DetectionMetrics(
            text="\n\n".join(valid_blocks),
            ai_probability=overall_ai,
            cyan_ai_score=round(sum(m.cyan_ai_score * m.word_count for m in block_metrics) / total_words, 1),
            purple_paraphrase_score=round(sum(m.purple_paraphrase_score * m.word_count for m in block_metrics) / total_words, 1),
            perplexity_proxy=round(sum(m.perplexity_proxy * m.word_count for m in block_metrics) / total_words, 2),
            burstiness=round(sum(m.burstiness * m.word_count for m in block_metrics) / total_words, 3),
            syntactic_symmetry=round(sum(m.syntactic_symmetry * m.word_count for m in block_metrics) / total_words, 3),
            synthetic_density=round(sum(m.synthetic_density * m.word_count for m in block_metrics) / total_words, 2),
            flagged_phrases=all_flagged,
            sentence_count=sum(m.sentence_count for m in block_metrics),
            word_count=total_words,
        )

    def analyze(self, text: str) -> dict:
        """Backward compatibility method for legacy callers."""
        metrics = self.score_text(text)
        return {
            "score": round(metrics.ai_probability / 100.0, 2),
            "matches": metrics.flagged_phrases,
            "sentence_variance": metrics.burstiness,
        }

AIDetector = AcademicAIDetector