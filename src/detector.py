import re
from typing import Dict, Any

class AIDetector:
    AI_BUZZWORDS = ["delve", "testament", "tapestry", "beacon", "vital", "leverage", "crucial", "moreover", "furthermore"]

    def analyze(self, text: str) -> Dict[str, Any]:
        found = [w for w in self.AI_BUZZWORDS if re.search(rf"\b{w}\b", text, re.I)]
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        lengths = [len(s.split()) for s in sentences]
        variance = 0.0
        if len(lengths) > 1:
            mean = sum(lengths) / len(lengths)
            variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        score = min(1.0, len(found) * 0.15 + (0.2 if variance < 10 and len(sentences) > 3 else 0.0))
        return {"score": round(score, 2), "matches": found, "sentence_variance": round(variance, 2)}
