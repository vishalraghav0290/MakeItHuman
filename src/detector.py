import re
from typing import Dict, Any, List

class AIDetector:
    """
    Heuristic-based detector for identifying common patterns and markers
    frequently found in AI-generated text.
    """

    # Common transition words and buzzwords disproportionately used by LLMs
    AI_BUZZWORDS = [
        "delve", "testament", "tapestry", "beacon", "vital",
        "leverage", "paradigm", "crucial", "underscore", "pivotal",
        "furthermore", "moreover", "in conclusion", "subsequently",
        "unwavering", "resonate", "interplay", "multifaceted"
    ]

    def __init__(self):
        # Pre-compile regex patterns for efficiency
        self.word_patterns = [
            re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
            for word in self.AI_BUZZWORDS
        ]

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the text for AI markers and calculates an estimated score.

        Returns:
            dict containing:
                - score (float): 0.0 (likely human) to 1.0 (likely AI)
                - matches (list): Identified buzzwords
                - sentence_variance (float): Metric representing sentence length diversity
        """
        if not text or not text.strip():
            return {"score": 0.0, "matches": [], "sentence_variance": 0.0}

        # Find buzzword occurrences
        found_words = []
        for word, pattern in zip(self.AI_BUZZWORDS, self.word_patterns):
            matches = pattern.findall(text)
            if matches:
                found_words.extend([word] * len(matches))

        # Analyze sentence structure and length variance (LLMs often have uniform burstiness)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_lengths = [len(s.split()) for s in sentences]

        variance = 0.0
        if len(sentence_lengths) > 1:
            mean = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((x - mean) ** 2 for x in sentence_lengths) / len(sentence_lengths)

        # Baseline heuristic calculation
        total_words = len(text.split())
        buzzword_density = len(found_words) / total_words if total_words > 0 else 0

        # High buzzword density + low variance in sentence length correlates with LLM drafts
        score = min(1.0, (buzzword_density * 15) + (0.2 if variance < 10 and len(sentences) > 3 else 0.0))

        return {
            "score": round(score, 2),
            "matches": list(set(found_words)),
            "sentence_variance": round(variance, 2)
        }