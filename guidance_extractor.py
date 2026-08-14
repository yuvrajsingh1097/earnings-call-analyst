"""
Forward Guidance Extractor
============================
Detects and scores forward-looking statements in earnings call transcripts.

Approach:
    1. Rule-based pattern matching  — catches explicit guidance phrases
    2. Tense + modal analysis       — identifies future-oriented language
    3. Guidance category classifier — revenue / margin / volume / capex / etc.
    4. Sentiment delta              — current vs prior quarter guidance sentiment
    5. Surprise score               — guidance sentiment vs analyst expectations

Guidance categories detected:
    revenue | margin | volume | capex | hiring | product | guidance_raise |
    guidance_lower | macro | competition
"""

import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Guidance trigger patterns
# ---------------------------------------------------------------------------

GUIDANCE_PATTERNS = [
    # Explicit forward guidance
    r"\bwe expect\b", r"\bwe anticipate\b", r"\bwe project\b",
    r"\bwe forecast\b", r"\bwe guide\b", r"\bour guidance\b",
    r"\bwe are guiding\b", r"\bgoing forward\b", r"\blooking ahead\b",
    r"\bfor the (next|coming|full|fiscal) (quarter|year|half)\b",
    r"\bin (q[1-4]|the (first|second|third|fourth) quarter)\b",
    r"\bfor fiscal (20\d\d)\b",
    # Outlook language
    r"\bwe (remain|are) (confident|optimistic|cautious|concerned)\b",
    r"\bwe (believe|see|feel) (the|our|that)\b.*\b(will|should|would)\b",
    r"\bwe (plan|intend|aim) to\b",
    r"\bwe will (continue|invest|focus|prioritize|accelerate)\b",
    r"\bwe (are on track|remain on track)\b",
    # Quantitative guidance
    r"\bexpect.{0,40}(grow|decline|increase|decrease|be)\b",
    r"\b(revenue|margin|growth|sales).{0,40}(expected|projected|anticipated)\b",
    r"\btargeting.{0,30}(percent|billion|million)\b",
]

GUIDANCE_RE = re.compile(
    "|".join(GUIDANCE_PATTERNS), flags=re.IGNORECASE
)

# Category keywords
CATEGORY_KEYWORDS = {
    "revenue":       ["revenue", "sales", "topline", "top-line", "bookings", "backlog"],
    "margin":        ["margin", "gross margin", "operating margin", "profitability", "ebitda"],
    "volume":        ["volume", "units", "shipments", "subscribers", "customers", "users"],
    "capex":         ["capex", "capital expenditure", "investment", "infrastructure", "capacity"],
    "hiring":        ["hiring", "headcount", "employees", "workforce", "talent"],
    "product":       ["product", "launch", "release", "feature", "innovation", "roadmap"],
    "guidance_raise":["raise", "raised", "increased guidance", "above", "exceed", "outperform"],
    "guidance_lower":["lower", "reduced", "cut", "below", "miss", "headwind", "pressure"],
    "macro":         ["macro", "economy", "inflation", "interest rate", "fx", "currency", "geopolitical"],
    "competition":   ["competition", "competitive", "market share", "competitor"],
}

# Positive / negative guidance sentiment words
GUIDANCE_POSITIVE = {
    "grow", "growth", "increase", "expand", "accelerate", "strong", "robust",
    "exceed", "outperform", "raise", "raised", "higher", "record", "momentum",
    "confident", "optimistic", "opportunity", "strength", "healthy", "sustained",
}
GUIDANCE_NEGATIVE = {
    "decline", "decrease", "lower", "reduce", "pressure", "headwind", "challenge",
    "cautious", "uncertain", "risk", "concern", "below", "miss", "weak", "soft",
    "difficult", "slowdown", "volatile", "unfavorable",
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GuidanceStatement:
    text: str
    speaker: str
    speaker_role: str
    category: str
    sentiment: str          # positive | negative | neutral
    sentiment_score: float  # [-1, 1]
    is_quantitative: bool   # contains a number / percentage
    quarter: str
    ticker: str


@dataclass
class GuidanceReport:
    ticker: str
    quarter: str
    statements: list = field(default_factory=list)
    overall_sentiment: float = 0.0
    sentiment_by_category: dict = field(default_factory=dict)
    n_quantitative: int = 0
    n_qualitative: int = 0
    guidance_surprise: Optional[float] = None  # vs analyst expectations


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class GuidanceExtractor:

    # Quantitative detection — numbers, percentages, ranges
    QUANT_RE = re.compile(
        r"\d+\.?\d*\s*(percent|%|billion|million|bps|basis points)|"
        r"\$\s*\d+\.?\d*|"
        r"\d+\s*to\s*\d+",
        flags=re.IGNORECASE
    )

    def extract_guidance_sentences(self, text: str) -> list:
        """
        Split text into sentences and return those containing guidance language.
        """
        # Simple sentence splitter
        sentences = re.split(r"(?<=[.!?])\s+", text)
        guidance = []
        for sent in sentences:
            if len(sent.split()) < 5:
                continue
            if GUIDANCE_RE.search(sent):
                guidance.append(sent.strip())
        return guidance

    def classify_category(self, text: str) -> str:
        """Classify guidance into a category based on keyword matching."""
        text_lower = text.lower()
        scores = {}
        for cat, keywords in CATEGORY_KEYWORDS.items():
            scores[cat] = sum(1 for kw in keywords if kw in text_lower)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def score_sentiment(self, text: str) -> tuple:
        """
        Score guidance sentiment using financial word lists.
        Returns (label, score) where score ∈ [-1, 1].
        """
        words = set(text.lower().split())
        pos = len(words & GUIDANCE_POSITIVE)
        neg = len(words & GUIDANCE_NEGATIVE)
        total = max(pos + neg, 1)

        score = (pos - neg) / total
        score = max(-1.0, min(1.0, score))

        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return label, round(score, 4)

    def is_quantitative(self, text: str) -> bool:
        """Check if a guidance statement contains specific numbers."""
        return bool(self.QUANT_RE.search(text))

    def extract(self, transcript) -> GuidanceReport:
        """
        Extract all guidance statements from a Transcript object.

        Returns GuidanceReport with all guidance statements,
        overall sentiment, and category breakdown.
        """
        statements = []

        for u in transcript.utterances:
            # Only management guidance (not analysts asking questions)
            if not u.speaker.is_management:
                continue
            if u.word_count < 8:
                continue

            guidance_sents = self.extract_guidance_sentences(u.text)

            for sent in guidance_sents:
                category = self.classify_category(sent)
                label, score = self.score_sentiment(sent)
                quant = self.is_quantitative(sent)

                statements.append(GuidanceStatement(
                    text=sent,
                    speaker=u.speaker.name,
                    speaker_role=u.speaker.role,
                    category=category,
                    sentiment=label,
                    sentiment_score=score,
                    is_quantitative=quant,
                    quarter=transcript.quarter,
                    ticker=transcript.ticker,
                ))

        # Aggregate
        overall = np.mean([s.sentiment_score for s in statements]) if statements else 0.0

        cat_sentiments = {}
        for cat in CATEGORY_KEYWORDS:
            cat_stmts = [s for s in statements if s.category == cat]
            if cat_stmts:
                cat_sentiments[cat] = round(
                    float(np.mean([s.sentiment_score for s in cat_stmts])), 4
                )

        n_quant = sum(1 for s in statements if s.is_quantitative)
        n_qual  = len(statements) - n_quant

        return GuidanceReport(
            ticker=transcript.ticker,
            quarter=transcript.quarter,
            statements=statements,
            overall_sentiment=round(float(overall), 4),
            sentiment_by_category=cat_sentiments,
            n_quantitative=n_quant,
            n_qualitative=n_qual,
        )


# ---------------------------------------------------------------------------
# Sentiment delta (quarter-over-quarter)
# ---------------------------------------------------------------------------

def compute_sentiment_delta(
    reports: list,
    ticker: str,
) -> pd.DataFrame:
    """
    Compute quarter-over-quarter guidance sentiment delta for a ticker.

    Parameters
    ----------
    reports : list of GuidanceReport objects (ordered by quarter)
    ticker  : ticker to filter

    Returns DataFrame with columns: quarter, sentiment, delta, signal
    """
    ticker_reports = [r for r in reports if r.ticker == ticker]
    if not ticker_reports:
        return pd.DataFrame()

    rows = []
    prev_sentiment = None
    for r in ticker_reports:
        delta = round(r.overall_sentiment - prev_sentiment, 4) \
                if prev_sentiment is not None else None
        rows.append({
            "ticker":    r.ticker,
            "quarter":   r.quarter,
            "sentiment": r.overall_sentiment,
            "delta":     delta,
            "signal":    "long" if (delta or 0) > 0.05
                         else "short" if (delta or 0) < -0.05
                         else "neutral",
        })
        prev_sentiment = r.overall_sentiment

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Guidance surprise score
# ---------------------------------------------------------------------------

def guidance_surprise_score(
    management_guidance_sentiment: float,
    analyst_question_sentiment: float,
) -> float:
    """
    Compute guidance surprise = management sentiment - analyst expectation proxy.

    Positive surprise → management more positive than analysts expected.
    Negative surprise → management more negative than analysts expected.

    Parameters
    ----------
    management_guidance_sentiment : compound sentiment of management's guidance
    analyst_question_sentiment    : compound sentiment of analyst questions
                                    (proxy for consensus expectation tone)
    """
    surprise = management_guidance_sentiment - analyst_question_sentiment
    return round(float(np.clip(surprise, -1, 1)), 4)


# ---------------------------------------------------------------------------
# Corpus-level summary
# ---------------------------------------------------------------------------

def guidance_corpus_summary(reports: list) -> pd.DataFrame:
    """
    Summarise guidance reports across all tickers and quarters.
    """
    rows = []
    for r in reports:
        rows.append({
            "ticker":           r.ticker,
            "quarter":          r.quarter,
            "n_statements":     len(r.statements),
            "n_quantitative":   r.n_quantitative,
            "n_qualitative":    r.n_qualitative,
            "overall_sentiment":r.overall_sentiment,
            **{f"cat_{k}": v for k, v in r.sentiment_by_category.items()},
        })
    return pd.DataFrame(rows).fillna(np.nan)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from scraper.transcript_loader import load_synthetic_transcript

    transcripts = [
        load_synthetic_transcript("AAPL"),
        load_synthetic_transcript("MSFT"),
    ]

    extractor = GuidanceExtractor()

    print("=" * 60)
    print("Forward Guidance Extractor Demo")
    print("=" * 60)

    reports = []
    for t in transcripts:
        report = extractor.extract(t)
        reports.append(report)

        print(f"\n[{report.ticker}] {report.quarter}")
        print(f"  Statements     : {len(report.statements)}")
        print(f"  Quantitative   : {report.n_quantitative}")
        print(f"  Qualitative    : {report.n_qualitative}")
        print(f"  Overall sent.  : {report.overall_sentiment:+.4f}")
        if report.sentiment_by_category:
            print(f"  By category    :")
            for cat, score in report.sentiment_by_category.items():
                print(f"    {cat:<18}: {score:+.4f}")

        print(f"\n  Sample statements:")
        for s in report.statements[:3]:
            print(f"    [{s.category}] {s.sentiment:>8} ({s.sentiment_score:+.3f}) "
                  f"{'[Q]' if s.is_quantitative else '   '}")
            print(f"    \"{s.text[:90]}...\"" if len(s.text) > 90 else f"    \"{s.text}\"")

    # Surprise score
    from nlp.finbert_scorer import FinBERTScorer
    scorer = FinBERTScorer(use_transformers=False)
    for t in transcripts:
        ts = scorer.score_transcript(t)
        mgmt_sent = ts.prepared_sentiment.compound
        anlst_sent = ts.analyst_sentiment.compound if ts.analyst_sentiment else 0
        surprise = guidance_surprise_score(mgmt_sent, anlst_sent)
        print(f"\n[{t.ticker}] Guidance surprise score: {surprise:+.4f}")

    # Summary
    summary = guidance_corpus_summary(reports)
    print("\nCorpus summary:")
    print(summary[["ticker", "quarter", "n_statements",
                   "overall_sentiment"]].to_string(index=False))
