import re
from typing import Dict, Any

class AIDetector:
    AI_BUZZWORDS = ["delve", "testament", "tapestry", "beacon", "vital", "leverage", "crucial", "moreover"]

    def analyze(self, text: str) -> Dict[str, Any]:
        found = [w for w in self.AI_BUZZWORDS if re.search(rf"\b{w}\b", text, re.I)]
        score = min(1.0, len(found) * 0.15)
        return {"score": round(score, 2), "matches": found}
