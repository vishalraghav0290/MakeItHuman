import re
import random
import logging

logger = logging.getLogger(__name__)

class TextHumanizer:
    """
    Transforms text by varying rhythm, breaking predictable AI phrases,
    and substituting robotic vocabulary with natural conversational variants.
    """

    # Replacement dictionary for commonly overused LLM terms
    VOCAB_MAPPING = {
        r"\bdelve into\b": ["look closely at", "explore", "dig into", "examine"],
        r"\bdelve\b": ["dig", "look", "jump into"],
        r"\ba testament to\b": ["proof of", "shows", "highlights"],
        r"\btapestry\b": ["mix", "blend", "combination"],
        r"\bpivotal\b": ["key", "critical", "important"],
        r"\bleverage\b": ["use", "make use of", "take advantage of"],
        r"\bmoreover\b": ["also", "on top of that", "plus"],
        r"\bfurthermore\b": ["also", "what's more", "in addition"],
        r"\bin conclusion\b": ["overall", "to wrap up", "all in all"],
        r"\bcrucial\b": ["important", "essential", "key"],
        r"\bvital\b": ["important", "needed", "essential"]
    }

    def __init__(self, seed: int = 42):
        # Optional seed for reproducible transformations if needed
        random.seed(seed)

    def humanize(self, text: str) -> str:
        """
        Executes rule-based humanization on the input text.
        """
        if not text:
            return ""

        result = text

        # 1. Substitute robotic phrasing with natural alternatives
        for pattern_str, alternatives in self.VOCAB_MAPPING.items():
            pattern = re.compile(pattern_str, re.IGNORECASE)
            
            def replace_match(match):
                word = random.choice(alternatives)
                # Keep capitalization if original was capitalized
                if match.group(0)[0].isupper():
                    return word.capitalize()
                return word

            result = pattern.sub(replace_match, result)

        # 2. Break uniform pacing where appropriate
        result = self._adjust_rhythm(result)

        return result

    def _adjust_rhythm(self, text: str) -> str:
        """
        Adjusts repetitive sentence structures and simplifies wordy qualifiers.
        """
        # Trim filler expressions
        fillers = [
            (re.compile(r"\bIt is important to note that\b", re.IGNORECASE), "Note that"),
            (re.compile(r"\bIn order to\b", re.IGNORECASE), "To"),
            (re.compile(r"\bDue to the fact that\b", re.IGNORECASE), "Because"),
        ]
        
        for pattern, replacement in fillers:
            text = pattern.sub(replacement, text)

        return text