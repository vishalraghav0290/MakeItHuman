import re

class TextHumanizer:
    VOCAB_MAPPING = {
        r"\bdelve into\b": "look into",
        r"\ba testament to\b": "proof of",
        r"\btapestry\b": "blend",
        r"\bleverage\b": "use",
    }

    def humanize(self, text: str) -> str:
        res = text
        for pat, repl in self.VOCAB_MAPPING.items():
            res = re.sub(pat, repl, res, flags=re.IGNORECASE)
        return res
