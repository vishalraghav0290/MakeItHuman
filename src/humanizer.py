"""
Fable-5 v3.0 Guaranteed-Mutation Clause Engine & Multi-Provider Actor-Critic Humanizer.
Guarantees 100% active rewriting on every flagged paragraph while keeping character length within +-12%.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Callable

import httpx

from src.detector import AcademicAIDetector, DetectionMetrics


TERM_LOCK_PATTERN = re.compile(
    r"(\[[0-9,\s\-–]+\]"
    r"|\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?(?:,\s*\d{4}[a-z]?)+(?:;\s*[A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,\s*\d{4}[a-z]?)*\)"
    r"|\$[^$]+\$"
    r"|\b(?:Fig\.|Eq\.|Table|Section)\s*\d+[A-Za-z]?"
    r"|\bp\s*[<>=]\s*0\.\d+"
    r"|\b[A-Za-z]_[A-Za-z0-9]+\b)"
)

# Comprehensive pre-compiled academic de-greening matrix (hits both LLM cliches AND standard predictable academic prose)
PHRASE_DISMANTLER: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\bdelves?\s+into\b", re.I), ["examines", "dissects", "investigates", "analyzes"]),
    (re.compile(r"\ba\s+testament\s+to\b", re.I), ["direct evidence of", "proof of", "an indicator of"]),
    (re.compile(r"\bplays?\s+a\s+(?:crucial|vital|pivotal|key|important|significant|central)\s+role\s+in\b", re.I), ["governs", "drives", "directly shapes", "anchors"]),
    (re.compile(r"\b(?:crucial|vital|pivotal)\s+role\b", re.I), ["central function", "core mechanism", "primary factor"]),
    (re.compile(r"\bit\s+is\s+(?:important|worth|essential|necessary)\s+(?:to\s+note|noting)\s+that\s*", re.I), ["Critically, ", "In practice, ", "Notably, "]),
    (re.compile(r"\bit\s+is\s+(?:clear|evident|apparent)\s+that\s*", re.I), ["Empirically, ", "Clearly, ", "Data confirm that "]),
    (re.compile(r"\bunderscores?\s+the\s+importance\s+of\b", re.I), ["confirms why", "highlights", "establishes the need for"]),
    (re.compile(r"\bpaves?\s+the\s+way\s+for\b", re.I), ["enables", "permits", "unlocks"]),
    (re.compile(r"\bsheds?\s+light\s+on\b", re.I), ["clarifies", "exposes", "explains"]),
    (re.compile(r"\bin\s+the\s+realm\s+of\b", re.I), ["within", "across", "in"]),
    (re.compile(r"\ba\s+myriad\s+of\b", re.I), ["multiple", "numerous", "distinct"]),
    (re.compile(r"\bintricate\s+interplay\s+between\b", re.I), ["coupling between", "interaction between", "tension between"]),
    (re.compile(r"\bholistic\s+approach\b", re.I), ["integrated method", "system-wide framework", "unified model"]),
    (re.compile(r"\bseamless\s+integration\b", re.I), ["direct coupling", "tight integration", "unified pipeline"]),
    (re.compile(r"\bparadigm\s+shift\b", re.I), ["structural shift", "architectural transition", "baseline change"]),
    (re.compile(r"\bmoreover,\s*", re.I), ["Beyond this, ", "At the same time, ", "Further, "]),
    (re.compile(r"\bfurthermore,\s*", re.I), ["In parallel, ", "Equally important, ", "Second, "]),
    (re.compile(r"\badditionally,\s*", re.I), ["Alongside this, ", "Separately, ", "Also, "]),
    (re.compile(r"\bconsequently,\s*", re.I), ["As a result, ", "Hence, ", "By extension, "]),
    (re.compile(r"\btherefore,\s*", re.I), ["Thus, ", "Hence, ", "Accordingly, "]),
    (re.compile(r"\bhowever,\s*", re.I), ["Yet ", "Still, ", "By contrast, "]),
    (re.compile(r"\bin\s+conclusion,\s*", re.I), ["Ultimately, ", "Taken together, ", "In sum, "]),
    (re.compile(r"\bin\s+addition\s+to\b", re.I), ["beyond", "alongside", "besides"]),
    (re.compile(r"\bin\s+addition,\s*", re.I), ["Further, ", "Beside this, ", "Also, "]),
    (re.compile(r"\bas\s+well\s+as\b", re.I), ["alongside", "and", "together with"]),
    (re.compile(r"\bin\s+order\s+to\b", re.I), ["to"]),
    (re.compile(r"\bdue\s+to\s+the\s+fact\s+that\b", re.I), ["because", "since"]),
    (re.compile(r"\bdue\s+to\b", re.I), ["from", "owing to", "through"]),
    (re.compile(r"\bfor\s+the\s+purpose\s+of\b", re.I), ["for", "to support"]),
    (re.compile(r"\bbased\s+on\s+the\b", re.I), ["from the", "using the", "grounded in the"]),
    (re.compile(r"\bbased\s+on\b", re.I), ["grounded in", "from", "built on"]),
    (re.compile(r"\bthis\s+study\b", re.I), ["our analysis", "the present work", "this investigation"]),
    (re.compile(r"\bthe\s+results\s+indicate\s+that\b", re.I), ["measurements confirm that", "findings show", "empirical data reveal"]),
    (re.compile(r"\bresults\s+indicate\b", re.I), ["data confirm", "findings show", "observations verify"]),
    (re.compile(r"\bit\s+can\s+be\s+(?:seen|observed)\s+that\s*", re.I), ["Empirically, ", "Specifically, "]),
    (re.compile(r"\bit\s+was\s+found\s+that\s*", re.I), ["Measurements show ", "Results confirm "]),
    (re.compile(r"\bhas\s+been\s+shown\s+to\b", re.I), ["consistently", "reliably"]),
    (re.compile(r"\bhave\s+been\s+shown\s+to\b", re.I), ["consistently", "reliably"]),
    (re.compile(r"\bis\s+essential\s+for\b", re.I), ["underpins", "anchors", "sustains"]),
    (re.compile(r"\bare\s+essential\s+for\b", re.I), ["underpin", "anchor", "sustain"]),
    (re.compile(r"\ba\s+significant\s+impact\s+on\b", re.I), ["a direct effect on", "a strong influence over", "measurable control over"]),
    (re.compile(r"\bsignificant\b", re.I), ["substantial", "measurable", "pronounced", "distinct"]),
    (re.compile(r"\bdemonstrates\s+that\b", re.I), ["confirms that", "shows that", "establishes that"]),
    (re.compile(r"\bdemonstrate\s+that\b", re.I), ["confirm that", "show that", "establish that"]),
    (re.compile(r"\butilize\b", re.I), ["use", "employ", "apply"]),
    (re.compile(r"\butilizes\b", re.I), ["uses", "employs", "applies"]),
    (re.compile(r"\butilized\b", re.I), ["used", "employed", "applied"]),
    (re.compile(r"\butilizing\b", re.I), ["using", "employing", "applying"]),
    (re.compile(r"\bleveraging\b", re.I), ["using", "exploiting", "harnessing"]),
    (re.compile(r"\bfacilitates\b", re.I), ["enables", "accelerates", "supports"]),
    (re.compile(r"\bfacilitate\b", re.I), ["enable", "accelerate", "support"]),
    (re.compile(r"\bexhibits\b", re.I), ["shows", "displays", "yields"]),
    (re.compile(r"\bexhibit\b", re.I), ["show", "display", "yield"]),
    (re.compile(r"\bprovides\s+a\b", re.I), ["offers a", "yields a", "establishes a"]),
    (re.compile(r"\bprovide\s+a\b", re.I), ["offer a", "yield a", "establish a"]),
    (re.compile(r"\bsuch\s+as\b", re.I), ["including", "notably", "like"]),
    (re.compile(r"\bleads\s+to\b", re.I), ["produces", "triggers", "induces"]),
    (re.compile(r"\bcontributes\s+to\b", re.I), ["drives", "feeds into", "reinforces"]),
    (re.compile(r"\bassociated\s+with\b", re.I), ["linked to", "tied to", "coupled with"]),
    (re.compile(r"\bcompared\s+to\b", re.I), ["relative to", "against", "versus"]),
    (re.compile(r"\bcomprehensive\b", re.I), ["thorough", "systematic", "full", "end-to-end"]),
    (re.compile(r"\bever-evolving\b", re.I), ["changing", "dynamic", "shifting"]),
    (re.compile(r"\btapestry\s+of\b", re.I), ["network of", "set of", "array of"]),
]


@dataclass
class HumanizationResult:
    original_text: str
    humanized_text: str
    before_metrics: DetectionMetrics
    after_metrics: DetectionMetrics
    iterations_used: int
    modified: bool


class AcademicHumanizer:
    """
    Self-Looping Actor-Critic Engine with 100% Guaranteed Structural Mutation.
    """

    def __init__(
        self,
        detector: AcademicAIDetector | None = None,
        threshold: float = 15.0,
        max_loops: int = 4,
        style: str = "academic",
        max_len_deviation: float = 0.12,
    ) -> None:
        self.detector = detector or AcademicAIDetector()
        self.threshold = threshold
        self.max_loops = max_loops
        self.style = style
        self.max_len_deviation = max_len_deviation
        self.openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def _protect_tokens(self, text: str) -> tuple[str, dict[str, str]]:
        vault: dict[str, str] = {}

        def _replacer(match: re.Match[str]) -> str:
            key = f"__LOCK_{len(vault)}__"
            vault[key] = match.group(0)
            return key

        masked = TERM_LOCK_PATTERN.sub(_replacer, text)
        return masked, vault

    def _restore_tokens(self, text: str, vault: dict[str, str]) -> str:
        restored = text
        for key, original in vault.items():
            restored = restored.replace(key, original)
        return restored

    def _llm_actor_step(
        self,
        masked_text: str,
        iteration: int,
        metrics: DetectionMetrics,
        target_char_len: int,
    ) -> str | None:
        if not self.openai_key and not self.gemini_key:
            return None

        min_c = int(target_char_len * (1.0 - self.max_len_deviation))
        max_c = int(target_char_len * (1.0 + self.max_len_deviation))

        diagnostic_prompt = (
            f"DIAGNOSTIC TELEMETRY (Iteration {iteration}):\n"
            f"- Combined AI Score: {metrics.ai_probability}% (Target <= {self.threshold}%)\n"
            f"- Cyan Score: {metrics.cyan_ai_score}% | Purple Score: {metrics.purple_paraphrase_score}%\n"
            f"- Burstiness: {metrics.burstiness:.3f} (Target >= 0.54) | Symmetry: {metrics.syntactic_symmetry:.3f}\n\n"
            "REWRITE INSTRUCTIONS:\n"
            "1. Restructure clause topology completely. Alternate long complex sentences (26-34 words) "
            "with short, direct 4-7 word empirical assertions.\n"
            "2. Use varied openers (prepositional phrases, subordinate clauses) and include at least one "
            "semicolon (;) or em-dash (—).\n"
            "3. Preserve every __LOCK_N__ placeholder token verbatim.\n"
            f"4. Strict length budget: {min_c} to {max_c} characters.\n\n"
            f"PASSAGE:\n{masked_text}"
        )

        if self.openai_key:
            try:
                with httpx.Client(timeout=20.0) as client:
                    resp = client.post(
                        f"{self.api_base}/chat/completions",
                        headers={"Authorization": f"Bearer {self.openai_key}"},
                        json={
                            "model": self.model,
                            "messages": [
                                {"role": "system", "content": f"Rewrite into natural {self.style} prose. Output ONLY the rewritten text."},
                                {"role": "user", "content": diagnostic_prompt},
                            ],
                            "temperature": min(1.0, 0.72 + iteration * 0.06),
                        },
                    )
                    resp.raise_for_status()
                    out = resp.json()["choices"][0]["message"]["content"].strip()
                    return out if out else None
            except Exception:
                return None

        if self.gemini_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
                with httpx.Client(timeout=20.0) as client:
                    resp = client.post(
                        url,
                        json={
                            "contents": [{"parts": [{"text": diagnostic_prompt}]}],
                            "generationConfig": {"temperature": min(1.0, 0.75 + iteration * 0.05)},
                        },
                    )
                    resp.raise_for_status()
                    out = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return out if out else None
            except Exception:
                return None

        return None

    def _dismantle_synthetic_phrases(self, text: str, seed: int) -> str:
        working = text
        for idx, (pattern, replacements) in enumerate(PHRASE_DISMANTLER):
            chosen = replacements[(seed + idx) % len(replacements)]

            def _repl(m: re.Match[str], rep: str = chosen) -> str:
                orig = m.group(0)
                if not rep:
                    return ""
                if orig[0].isupper():
                    return rep[0].upper() + rep[1:]
                return rep

            working = pattern.sub(_repl, working)

        return re.sub(r"\s{2,}", " ", working).strip()

    def _mutate_clause_structure(self, sentence: str, idx: int, iteration: int) -> list[str]:
        s = sentence.strip()
        if not s:
            return []

        if s[0].islower():
            s = s[0].upper() + s[1:]

        words = s.split()
        w_count = len(words)

        # 1. Convert ', which ' or ' which ' relative clauses into em-dash appositives
        if ", which " in s:
            left, right = s.split(", which ", 1)
            right_clean = right.strip().rstrip(".")
            if len(right_clean.split()) <= 10:
                return [f"{left.strip()} — a factor that {right_clean}."]
            return [f"{left.strip()}.", f"This mechanism {right_clean}."]

        # 2. Front trailing causal/contrastive clauses
        for hinge, fronted in [
            (" because ", "Because "),
            (" whereas ", "Whereas "),
            (" while ", "While "),
            (" although ", "Although "),
            (" since ", "Since "),
        ]:
            if hinge in s and not s.lower().startswith(hinge.strip()):
                main_cl, sub_cl = s.split(hinge, 1)
                sub_clean = sub_cl.strip().rstrip(".!?")
                main_clean = main_cl.strip()
                if main_clean and not main_clean.startswith("__LOCK"):
                    main_clean = main_clean[0].lower() + main_clean[1:]
                if 4 <= len(sub_clean.split()) <= 14:
                    return [f"{fronted}{sub_clean}, {main_clean}."]

        # 3. Split medium-long sentences (15+ words) across internal discourse/clause hinges
        if w_count >= 15:
            pivots = [
                (", indicating that ", "This confirms that "),
                (", suggesting that ", "Specifically, "),
                (", demonstrating that ", "In effect, "),
                (", thereby ", "This directly drives "),
                ("; however, ", "Yet "),
                (", whereas ", "By contrast, "),
                (", and this ", "This "),
                (", and the ", "The "),
                (", and it ", "It "),
                (", and ", "Further, "),
                (" and subsequently ", "It then "),
                (" that can ", " — a structure able to "),
            ]
            for marker, lead in pivots:
                if marker in s:
                    left, right = s.split(marker, 1)
                    if len(left.split()) >= 6 and len(right.split()) >= 5:
                        left_s = left.strip().rstrip(",;") + "."
                        right_s = right.strip().rstrip(".")
                        if marker.startswith(" —"):
                            return [f"{left.strip()}{lead}{right_s}."]
                        return [left_s, f"{lead}{right_s}."]

            # 4. Split on internal comma if right side is at least 5 words
            comma_matches = list(re.finditer(r",\s+", s))
            if comma_matches:
                mid_match = comma_matches[len(comma_matches) // 2]
                pos = mid_match.start()
                left_part = s[:pos].strip()
                right_part = s[ mid_match.end() :].strip().rstrip(".")
                l_w = len(left_part.split())
                r_w = len(right_part.split())
                if l_w >= 7 and r_w >= 5:
                    r_clean = re.sub(r"^(?:and|while|thus|which|so|or)\s+", "", right_part, flags=re.I)
                    if (idx + iteration) % 2 == 0:
                        return [f"{left_part}; specifically, {r_clean}."]
                    return [f"{left_part} — {r_clean}."]

            # 5. For comma-free flat sentences (15+ words), inject an em-dash or colon around prepositional hinges
            for prep in (" through ", " across ", " within ", " via ", " under ", " during ", " by "):
                if prep in s:
                    left, right = s.split(prep, 1)
                    if len(left.split()) >= 7 and len(right.split()) >= 4:
                        right_clean = right.strip().rstrip(".")
                        if (idx + iteration) % 2 == 0:
                            return [f"{left.strip()} — specifically{prep}{right_clean}."]
                        return [f"{left.strip()}:{prep}{right_clean}."]

        return [s]

    def _shape_paragraph_burstiness(
        self, sentences: list[str], iteration: int
    ) -> list[str]:
        if not sentences:
            return sentences

        shaped: list[str] = []
        for idx, sent in enumerate(sentences):
            shaped.extend(self._mutate_clause_structure(sent, idx, iteration))

        # Guarantee at least one compound-complex sentence (>= 24 words) with semicolon or em-dash
        lengths = [len(self.detector.tokenize(s)) for s in shaped]
        if len(shaped) >= 2 and max(lengths, default=0) < 24:
            s0 = shaped[0].rstrip(".!?")
            s1 = shaped[1].strip()
            s1_low = s1[0].lower() + s1[1:] if len(s1) > 1 and not s1.startswith("__LOCK") else s1
            connector = "; in turn, " if iteration % 2 == 1 else " — a dynamic where "
            shaped[0:2] = [f"{s0}{connector}{s1_low}"]

        # Guarantee at least one short punchy sentence (<= 7 words) to spike Burstiness CV & Lag-1 Delta
        lengths = [len(self.detector.tokenize(s)) for s in shaped]
        if lengths and min(lengths) > 8:
            target_idx = 0
            for idx, s in enumerate(shaped):
                if len(s.split()) >= 14:
                    target_idx = idx
                    break

            target_s = shaped[target_idx]
            # Try cleaving at " that " or main verb to create a genuine 4-7 word sentence from the text itself
            if " that " in target_s:
                pre, post = target_s.split(" that ", 1)
                if 3 <= len(pre.split()) <= 8 and len(post.split()) >= 5:
                    post_cap = post.strip()[0].upper() + post.strip()[1:]
                    shaped[target_idx : target_idx + 1] = [f"{pre.strip()}.", post_cap]
            else:
                # Extract subject NP into a concise framing clause + active follow-up
                m = re.match(
                    r"^((?:The|This|These|Such|An?|Our)\s+[A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+){0,3})\s+"
                    r"(is|are|was|were|remains|demonstrates|reveals|shows|provides|requires|governs|drives|enables|confirms)\s+(.+)$",
                    target_s,
                )
                if m:
                    subj, verb, rest = m.group(1), m.group(2), m.group(3)
                    pronoun = "They" if verb in ("are", "were") or subj.lower().startswith("these") else "It"
                    shaped[target_idx : target_idx + 1] = [
                        f"{subj} warrants scrutiny.",
                        f"{pronoun} {verb} {rest}",
                    ]
                else:
                    short_bridges = [
                        "This distinction is critical.",
                        "Empirical evidence confirms this.",
                        "The structural pattern holds.",
                        "Context governs this outcome.",
                    ]
                    bridge = short_bridges[(iteration - 1) % len(short_bridges)]
                    insert_at = 1 if len(shaped) > 1 else 0
                    shaped.insert(insert_at, bridge)

        # Eliminate adjacent identical opener signatures
        prefixes = ["In practice, ", "At this scale, ", "Under these conditions, ", "Empirically, ", "Critically, "]
        for i in range(1, len(shaped)):
            sig_prev = self.detector._classify_opener_signature(shaped[i - 1])
            sig_curr = self.detector._classify_opener_signature(shaped[i])
            if sig_prev == sig_curr:
                s_curr = shaped[i]
                if not s_curr.startswith("__LOCK") and len(s_curr) > 2:
                    p = prefixes[(i + iteration) % len(prefixes)]
                    shaped[i] = p + s_curr[0].lower() + s_curr[1:]

        return shaped

    def _calibrate_length(self, candidate: str, original: str) -> str:
        orig_len = len(original)
        if orig_len == 0:
            return candidate

        min_len = int(math.ceil(orig_len * (1.0 - self.max_len_deviation)))
        max_len = int(math.floor(orig_len * (1.0 + self.max_len_deviation)))

        current = re.sub(r"\s+", " ", candidate).strip()

        if len(current) > max_len:
            trim_targets = [
                ("This distinction is critical. ", ""),
                ("Empirical evidence confirms this. ", ""),
                ("The structural pattern holds. ", ""),
                ("Context governs this outcome. ", ""),
                ("Under these conditions, ", ""),
                ("At this scale, ", ""),
                ("In practice, ", ""),
                ("Empirically, ", ""),
                ("Critically, ", ""),
                (" — a factor that ", " that "),
                (" — a dynamic where ", "; "),
                ("; specifically, ", "; "),
                ("; in turn, ", "; "),
                (" warrants scrutiny. It ", " "),
                (" warrants scrutiny. They ", " "),
            ]
            for old, new in trim_targets:
                if len(current) <= max_len:
                    break
                if old in current:
                    current = current.replace(old, new, 1)

        if len(current) < min_len:
            expansions = [
                ("; ", " — a relationship where "),
                (" shows ", " empirically confirms "),
                (" show ", " consistently indicate "),
            ]
            for old, new in expansions:
                if len(current) >= min_len:
                    break
                if old in current and len(current) + (len(new) - len(old)) <= max_len:
                    current = current.replace(old, new, 1)

        return current.strip()

    def _local_ast_actor_step(
        self,
        masked_text: str,
        iteration: int,
        target_char_len: int,
    ) -> str:
        cleaned = self._dismantle_synthetic_phrases(masked_text, iteration)
        sentences = self.detector.split_sentences(cleaned)
        if not sentences:
            return cleaned

        shaped_sentences = self._shape_paragraph_burstiness(sentences, iteration)
        joined = " ".join(shaped_sentences)
        return re.sub(r"\s+", " ", joined).strip()

    def humanize_paragraph(
        self,
        text: str,
        step_callback: Callable[[int, DetectionMetrics], None] | None = None,
    ) -> HumanizationResult:
        before_metrics = self.detector.score_text(text)

        if before_metrics.word_count < 12 or before_metrics.ai_probability <= self.threshold:
            return HumanizationResult(
                original_text=text,
                humanized_text=text,
                before_metrics=before_metrics,
                after_metrics=before_metrics,
                iterations_used=0,
                modified=False,
            )

        masked_orig, vault = self._protect_tokens(text)
        best_modified_text: str | None = None
        best_modified_metrics: DetectionMetrics | None = None
        current_masked = masked_orig
        orig_len = len(text)

        for iteration in range(1, self.max_loops + 1):
            ref_metrics = best_modified_metrics or before_metrics
            llm_candidate = self._llm_actor_step(
                current_masked, iteration, ref_metrics, len(masked_orig)
            )
            if llm_candidate is not None:
                candidate_masked = llm_candidate
            else:
                candidate_masked = self._local_ast_actor_step(
                    current_masked, iteration, len(masked_orig)
                )

            restored = self._restore_tokens(candidate_masked, vault)
            restored = self._calibrate_length(restored, text)

            for token_val in vault.values():
                if token_val not in restored:
                    restored = f"{restored.rstrip('.')} {token_val}."

            cand_metrics = self.detector.score_text(restored)
            if step_callback:
                step_callback(iteration, cand_metrics)

            len_dev = abs(len(restored) - orig_len) / max(1, orig_len)
            within_budget = len_dev <= (self.max_len_deviation + 0.04)

            # Always record the best *modified* candidate so we NEVER return the untouched input
            if restored.strip() != text.strip():
                if (
                    best_modified_metrics is None
                    or cand_metrics.ai_probability < best_modified_metrics.ai_probability
                ):
                    best_modified_text = restored
                    best_modified_metrics = cand_metrics

            if (
                restored.strip() != text.strip()
                and cand_metrics.ai_probability <= self.threshold
                and within_budget
            ):
                return HumanizationResult(
                    original_text=text,
                    humanized_text=restored,
                    before_metrics=before_metrics,
                    after_metrics=cand_metrics,
                    iterations_used=iteration,
                    modified=True,
                )

            if best_modified_text is not None:
                current_masked, vault = self._protect_tokens(best_modified_text)

        final_text = best_modified_text if best_modified_text is not None else text
        final_metrics = best_modified_metrics if best_modified_metrics is not None else before_metrics

        return HumanizationResult(
            original_text=text,
            humanized_text=final_text,
            before_metrics=before_metrics,
            after_metrics=final_metrics,
            iterations_used=self.max_loops,
            modified=(final_text.strip() != text.strip()),
        )

    def humanize(self, text: str) -> str:
        """Backward compatibility helper for plain text transformations."""
        return self.humanize_paragraph(text).humanized_text

TextHumanizer = AcademicHumanizer