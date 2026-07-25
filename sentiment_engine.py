"""
sentiment_engine.py
--------------------
Core NLP logic for the AI Sentiment Analysis teaching app.

This module intentionally contains TWO different sentiment approaches so
students can compare them side by side:

1. SimpleLexiconSentiment
   A transparent, from-scratch, rule-based analyzer. It shows every step
   of a classic NLP sentiment pipeline: tokenization -> lexicon lookup ->
   negation handling -> intensifier scaling -> aggregation. Nothing here
   is a "black box" - it is meant to be read by students.

2. VaderSentiment
   A thin wrapper around NLTK's VADER (Valence Aware Dictionary and
   sEntiment Reasoner), a well-known, production-grade lexicon and
   rule-based model that is widely used in industry and research for
   short, informal text (tweets, reviews, comments).

Both analyzers expose the same output shape (a dict) so the Streamlit UI
can treat them interchangeably.
"""

import re
import string
from dataclasses import dataclass, field
from typing import List, Dict


# ---------------------------------------------------------------------------
# 1. SIMPLE LEXICON-BASED SENTIMENT ANALYZER (from scratch, for teaching)
# ---------------------------------------------------------------------------

# A deliberately small, readable lexicon. In a real system this would be
# thousands of words (e.g. AFINN, Bing Liu's opinion lexicon, SentiWordNet).
POSITIVE_WORDS: Dict[str, float] = {
    "good": 1.5, "great": 2.0, "excellent": 2.5, "amazing": 2.5, "awesome": 2.5,
    "love": 2.0, "like": 1.0, "happy": 1.8, "wonderful": 2.2, "fantastic": 2.3,
    "best": 2.0, "beautiful": 1.8, "nice": 1.2, "perfect": 2.4, "enjoy": 1.5,
    "enjoyed": 1.5, "recommend": 1.5, "impressive": 1.8, "brilliant": 2.2,
    "helpful": 1.3, "delight": 1.8, "delightful": 1.9, "pleased": 1.5,
    "fun": 1.4, "easy": 1.0, "smooth": 1.2, "reliable": 1.3, "fast": 1.0,
}

NEGATIVE_WORDS: Dict[str, float] = {
    "bad": -1.5, "terrible": -2.5, "awful": -2.5, "horrible": -2.5,
    "hate": -2.2, "dislike": -1.3, "poor": -1.5, "worst": -2.5,
    "disappointing": -1.8, "disappointed": -1.8, "broken": -1.5,
    "useless": -1.8, "slow": -1.0, "confusing": -1.3, "annoying": -1.5,
    "waste": -1.6, "boring": -1.3, "difficult": -1.0, "problem": -1.2,
    "problems": -1.2, "issue": -1.0, "issues": -1.0, "buggy": -1.6,
    "crash": -1.8, "crashes": -1.8, "expensive": -1.0, "fail": -1.8,
    "failed": -1.8,
}

NEGATION_WORDS = {
    "not", "no", "never", "none", "n't", "cannot", "can't", "won't",
    "isn't", "wasn't", "aren't", "weren't", "don't", "doesn't", "didn't",
}

INTENSIFIERS: Dict[str, float] = {
    "very": 1.5, "extremely": 1.8, "really": 1.3, "so": 1.3,
    "incredibly": 1.7, "absolutely": 1.6, "totally": 1.4, "quite": 1.2,
}

DIMINISHERS: Dict[str, float] = {
    "slightly": 0.6, "somewhat": 0.7, "barely": 0.5, "kind of": 0.7,
    "a bit": 0.7,
}


@dataclass
class WordContribution:
    """Records how a single token affected the final score, for teaching
    purposes (so the UI can highlight words and show the 'why')."""
    word: str
    base_score: float
    negated: bool = False
    scaled_by: float = 1.0
    final_score: float = 0.0


def tokenize(text: str) -> List[str]:
    """A minimal tokenizer: lowercase, then split on whitespace after
    stripping most punctuation (but keeping apostrophes so 'don't' survives
    as one token, which matters for negation detection)."""
    text = text.lower()
    # Remove punctuation except apostrophes
    keep_apostrophe = str.maketrans("", "", string.punctuation.replace("'", ""))
    text = text.translate(keep_apostrophe)
    tokens = text.split()
    return tokens


class SimpleLexiconSentiment:
    """A transparent, hand-rolled rule-based sentiment analyzer.

    Pipeline:
        1. Tokenize
        2. For each token, look up base sentiment score (if any)
        3. Check the previous 1-2 tokens for negation -> flip sign
        4. Check the previous token for an intensifier/diminisher -> scale
        5. Sum all token scores
        6. Normalize into a -1..+1 "compound-like" score and a label
    """

    NEGATION_WINDOW = 3  # how many tokens back we look for a negator

    def analyze(self, text: str) -> Dict:
        tokens = tokenize(text)
        contributions: List[WordContribution] = []
        raw_total = 0.0

        for i, token in enumerate(tokens):
            base = POSITIVE_WORDS.get(token) or NEGATIVE_WORDS.get(token)
            if base is None:
                continue

            window_start = max(0, i - self.NEGATION_WINDOW)
            preceding = tokens[window_start:i]

            negated = any(w in NEGATION_WORDS for w in preceding)

            scale = 1.0
            if i > 0 and tokens[i - 1] in INTENSIFIERS:
                scale = INTENSIFIERS[tokens[i - 1]]
            elif i > 0 and tokens[i - 1] in DIMINISHERS:
                scale = DIMINISHERS[tokens[i - 1]]

            score = base * scale
            if negated:
                score *= -1.0

            contributions.append(WordContribution(
                word=token, base_score=base, negated=negated,
                scaled_by=scale, final_score=score,
            ))
            raw_total += score

        # Normalize roughly into [-1, 1] using a soft cap, similar in spirit
        # to how VADER normalizes its compound score.
        if raw_total == 0:
            normalized = 0.0
        else:
            normalized = raw_total / (abs(raw_total) + 3.0)
            normalized = max(-1.0, min(1.0, normalized))

        label = self._label(normalized)

        return {
            "method": "Simple Lexicon (from scratch)",
            "text": text,
            "tokens": tokens,
            "contributions": contributions,
            "raw_score": raw_total,
            "compound": normalized,
            "label": label,
        }

    @staticmethod
    def _label(compound: float) -> str:
        if compound >= 0.2:
            return "Positive"
        elif compound <= -0.2:
            return "Negative"
        return "Neutral"


# ---------------------------------------------------------------------------
# 2. VADER WRAPPER (industry-standard lexicon + rule-based model via NLTK)
# ---------------------------------------------------------------------------

class VaderSentiment:
    """Wraps NLTK's VADER SentimentIntensityAnalyzer.

    VADER (Hutto & Gilbert, 2014) is a lexicon-and-rule-based model tuned
    for social-media-style text. It returns four scores:
        neg, neu, pos  -> proportions of the text that are negative,
                          neutral, and positive (they sum to ~1.0)
        compound       -> a single normalized score in [-1, 1] that
                          aggregates the whole sentence, computed by
                          summing valence scores of each word (with
                          heuristics for punctuation, capitalization,
                          intensifiers, and negation), then normalizing:
                              compound = x / sqrt(x^2 + alpha)
                          where alpha = 15 is a normalization constant.
    """

    _analyzer = None

    def __init__(self):
        if VaderSentiment._analyzer is None:
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            VaderSentiment._analyzer = SentimentIntensityAnalyzer()
        self.analyzer = VaderSentiment._analyzer

    def analyze(self, text: str) -> Dict:
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]
        label = self._label(compound)
        return {
            "method": "VADER (NLTK)",
            "text": text,
            "neg": scores["neg"],
            "neu": scores["neu"],
            "pos": scores["pos"],
            "compound": compound,
            "label": label,
        }

    @staticmethod
    def _label(compound: float) -> str:
        # These exact thresholds are VADER's documented defaults.
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        return "Neutral"


def get_analyzer(method: str):
    """Factory function so the UI layer doesn't need to know construction
    details of each analyzer."""
    if method == "vader":
        return VaderSentiment()
    return SimpleLexiconSentiment()
