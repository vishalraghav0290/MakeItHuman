"""
Turnitin & ZeroGPT Calibrated N-Gram Predictability & Burstiness Detector (v4.0).
Accurately flags 1-sentence bullet points and short project-report paragraphs.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field


LLM_LEXICAL_WEIGHTS: dict[str, float] = {
    "delve into": 4.5, "delves into": 4.5, "testament to": 4.2, "crucial role": 3.5,
    "vital role": 3.2, "pivotal role": 3.5, "important role": 2.8, "key role": 2.6,
    "moreover": 2.8, "furthermore": 2.8, "consequently": 2.2, "additionally": 2.2,
    "therefore": 1.8, "however": 1.5, "nonetheless": 2.4, "nevertheless": 2.4,
    "it is important to note": 4.0, "it is worth noting": 3.8, "tapestry": 4.8,
    "intricate interplay": 4.6, "shed light on": 3.4, "pave the way": 3.5,
    "underscores": 3.0, "in conclusion": 2.6, "a myriad of": 3.9, "paradigm shift": 3.4,
    "seamless integration": 3.8, "seamlessly": 3.2, "holistic approach": 3.5,
    "holistic": 2.8, "groundbreaking": 3.1, "in the realm of": 3.9, "meticulously": 3.6,
    "ever-evolving": 4.0, "multifaceted": 3.3, "nuanced": 2.5, "comprehensive": 2.4,
    "notably": 2.0, "significantly": 1.8, "demonstrates": 1.8, "exhibits": 1.8,
    "facilitates": 2.4, "facilitate": 2.4, "utilizing": 2.5, "utilize": 2.5,
    "utilizes": 2.5, "leveraging": 3.0, "leverage": 2.8, "synergy": 3.4,
    "cornerstone": 3.3, "paramount": 3.0, "poised to": 3.5, "landscape": 2.5,
    "robust": 2.2, "scalable": 2.0, "streamline": 2.6, "streamlines": 2.6,
    "empower": 2.8, "empowers": 2.8, "enhance": 2.0, "enhances": 2.0, "enhancing": 2.0,
    "optimize": 2.0, "optimizing": 2.0, "ensure": 1.8, "ensures": 1.8, "ensuring": 1.8,
    "provides a": 1.8, "allows for": 2.0, "enables": 1.8, "designed to": 1.9,
    "aims to": 2.0, "focuses on": 1.8, "in order to": 2.0, "due to the": 1.8,
    "as well as": 1.7, "such as": 1.5, "based on": 1.6, "associated with": 1.8,
}

COMPILED_LEXICAL_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    (phrase, re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE), weight)
    for phrase, weight in LLM_LEXICAL_WEIGHTS.items()
]

# High-frequency GPT token bigrams that trigger ZeroGPT / Turnitin even in short 1-sentence blocks
PREDICTABLE_BIGRAMS: set[tuple[str, str]] = {
    ("this", "project"), ("the", "system"), ("system", "is"), ("is", "designed"),
    ("designed", "to"), ("to", "provide"), ("provides", "a"), ("allows", "users"),
    ("users", "to"), ("in", "order"), ("order", "to"), ("as", "well"), ("well", "as"),
    ("such", "as"), ("based", "on"), ("can", "be"), ("it", "is"), ("of", "the"),
    ("to", "the"), ("in", "the"), ("for", "the"), ("with", "the"), ("on", "the"),
    ("by", "the"), ("from", "the"), ("and", "the"), ("that", "the"), ("is", "a"),
    ("are", "used"), ("is", "used"), ("used", "to"), ("able", "to"), ("helps", "to"),
    ("helps", "in"), ("ensures", "that"), ("ensure", "that"), ("focuses", "on"),
    ("aims", "to"), ("plays", "a"), ("a", "key"), ("a", "crucial"), ("a", "vital"),
    ("a", "comprehensive"), ("a", "robust"), ("a", "seamless"), ("user", "experience"),
    ("real", "time"), ("data", "management"), ("efficient", "and"), ("and", "effective"),
    ("and", "reliable"), ("and", "secure"), ("and", "scalable"), ("by", "utilizing"),
    ("by", "leveraging"), ("through", "the"), ("across", "various"), ("various", "modules"),
}

ABBREV_PROTECT_REGEX = re.compile(
    r"\b(et\s+al|e\.g|i\.e|Fig|Figs|Eq|Eqs|Dr|Prof|Mr|Mrs|Ms|vs|etc|No|Vol|pp|ch|sec)\.",
    re.IGNORECASE,
)
SENTENCE_BOUNDARY_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"\'])")
WORD_REGEX = re.compile(r"\b[a-zA-Z0-9'-]+\b")
HEADING_OR_NOISE_REGEX = re.compile(
    r"^(?:chapter\s+\d+|section\s+\d+|table\s+of\s+contents|list\s+of\s+(?:figures|tables)|"
    r"references|bibliography|appendix|abstract|acknowledg(?:e)?ments|declaration|certificate|"
    r"\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9\s\-–:&/]{1,65}|"
    r".*\.{4,}\s*\d+)$",
    re.IGNORECASE,
)


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
    """Evaluates both multi-sentence paragraphs and single-sentence project report blocks."""

    def is_heading_or_metadata(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        words = stripped.split()
        if len(words) < 6:
            return True
        # Skip TOC lines, numbered headings without terminal periods, code lines, URLs
        if HEADING_OR_NOISE_REGEX.match(stripped):
            return True
        if len(words) <= 10 and not stripped.endswith((".", "!", "?", ":", ";")):
            return True
        if stripped.isupper() and len(words) <= 14:
            return True
        return False

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
        if len(tokens) < 6:
            return 75.0

        bigrams = list(zip(tokens[:-1], tokens[1:]))
        predictable_hits = sum(1 for bg in bigrams if bg in PREDICTABLE_BIGRAMS)
        predictable_ratio = predictable_hits / max(1, len(bigrams))

        counts = Counter(tokens)
        hapax_ratio = sum(1 for c in counts.values() if c == 1) / len(tokens)

        word_lengths = [len(t) for t in tokens]
        len_std = statistics.stdev(word_lengths) if len(word_lengths) > 2 else 2.0

        semi_colon_dash = len(re.findall(r"[;:—–\(\)\?]", text))
        word_scale = max(0.5, len(tokens) / 50.0)
        punct_bonus = min(20.0, (semi_colon_dash / word_scale) * 4.5)

        surprisal = (
            (hapax_ratio * 55.0)
            + (len_std * 7.0)
            + punct_bonus
            - (predictable_ratio * 165.0)
        )
        return round(max(5.0, min(98.0, surprisal)), 2)

    def calculate_burstiness_and_pacing(self, sentences: list[str]) -> tuple[float, float]:
        lengths = [len(self.tokenize(s)) for s in sentences if len(self.tokenize(s)) > 0]
        if not lengths:
            return 0.50, 0.20

        # Single-sentence blocks in AI reports almost always have 13-28 words with flat cadence
        if len(lengths) == 1:
            L = lengths[0]
            if 12 <= L <= 30:
                return 0.22, 0.85
            return 0.45, 0.35

        mean_len = statistics.mean(lengths)
        stdev_len = statistics.stdev(lengths)
        cv = stdev_len / max(1.0, mean_len)

        lag_diffs = [abs(lengths[i] - lengths[i - 1]) for i in range(1, len(lengths))]
        mean_lag_diff = statistics.mean(lag_diffs) / max(1.0, mean_len)

        in_ai_band = sum(1 for L in lengths if 13 <= L <= 28)
        band_concentration = in_ai_band / len(lengths)

        composite_burstiness = (0.55 * cv) + (0.45 * mean_lag_diff)
        return round(composite_burstiness, 4), round(band_concentration, 4)

    def calculate_syntactic_symmetry(self, sentences: list[str]) -> float:
        if len(sentences) == 1:
            sig = self._classify_opener_signature(sentences[0])
            return 0.65 if sig in ("DET_NP", "CONJ_ADV", "PRON_SUBJ") else 0.25

        if len(sentences) == 2:
            s0 = self._classify_opener_signature(sentences[0])
            s1 = self._classify_opener_signature(sentences[1])
            return 0.72 if s0 == s1 else 0.32

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
        return round(density_score, 3), 0.0, flagged

    def score_text(self, text: str) -> DetectionMetrics:
        if self.is_heading_or_metadata(text):
            return DetectionMetrics(
                text=text, ai_probability=0.0, cyan_ai_score=0.0, purple_paraphrase_score=0.0,
                perplexity_proxy=90.0, burstiness=0.65, syntactic_symmetry=0.15,
                synthetic_density=0.0, flagged_phrases=[], sentence_count=1,
                word_count=len(self.tokenize(text)),
            )

        sentences = self.split_sentences(text)
        tokens = self.tokenize(text)
        word_count = len(tokens)

        perplexity_proxy = self.calculate_perplexity_proxy(tokens, text)
        burstiness, band_concentration = self.calculate_burstiness_and_pacing(sentences)
        symmetry = self.calculate_syntactic_symmetry(sentences)
        synth_density, _, flagged = self.calculate_synthetic_and_purple_risk(text, tokens)

        burst_deficit = max(0.0, 0.56 - burstiness) * 6.5
        band_penalty = max(0.0, band_concentration - 0.38) * 3.8
        perp_deficit = max(0.0, (66.0 - perplexity_proxy) / 11.5)
        sym_penalty = max(0.0, (symmetry - 0.30) * 4.2)
        lex_penalty = min(5.0, synth_density * 0.32 + len(flagged) * 0.65)

        logit_z = -1.85 + burst_deficit + band_penalty + perp_deficit + sym_penalty + lex_penalty
        cyan_prob = 100.0 / (1.0 + math.exp(-logit_z))

        purple_score = max(
            0.0,
            min(99.0, (band_concentration * 38.0) + (symmetry * 35.0) - max(0.0, burstiness - 0.48) * 65.0),
        )
        combined_prob = round(max(1.0, min(99.4, max(cyan_prob, cyan_prob * 0.7 + purple_score * 0.35))), 1)

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
        valid_blocks = [b.strip() for b in blocks if not self.is_heading_or_metadata(b)]
        if not valid_blocks:
            return self.score_text(" ".join(blocks))

        windows: list[str] = []
        current_group: list[str] = []
        current_words = 0

        for block in valid_blocks:
            w_len = len(self.tokenize(block))
            current_group.append(block)
            current_words += w_len
            if current_words >= 200:
                windows.append(" ".join(current_group))
                current_group = [current_group[-1]] if len(current_group) > 1 else []
                current_words = len(self.tokenize(current_group[0])) if current_group else 0

        if current_group:
            windows.append(" ".join(current_group))

        window_metrics = [self.score_text(w) for w in windows]
        block_metrics = [self.score_text(b) for b in valid_blocks]

        total_words = max(1, sum(m.word_count for m in block_metrics))
        total_win_words = max(1, sum(m.word_count for m in window_metrics))

        accum = sum(m.ai_probability * m.word_count for m in window_metrics)
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
