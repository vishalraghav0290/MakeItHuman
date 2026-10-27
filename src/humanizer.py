import re

class TextHumanizer:
    VOCAB_MAPPING = {
        r"\bdelve into\b": "look closely at",
        r"\ba testament to\b": "proof of",
        r"\btapestry\b": "mix",
        r"\bleverage\b": "use",
        r"\bmoreover\b": "also",
        r"\bfurthermore\b": "plus",
    }

    def humanize(self, text: str) -> str:
        res = text
        for pat, repl in self.VOCAB_MAPPING.items():
            res = re.sub(pat, repl, res, flags=re.IGNORECASE)
        res = re.sub(r"\bIn order to\b", "To", res, flags=re.IGNORECASE)
        return res
