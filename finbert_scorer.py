"""
FinBERT Sentiment Scorer
=========================
Scores earnings call utterances using FinBERT — a BERT model
fine-tuned on financial text (ProsusAI/finbert on HuggingFace).

Labels:  positive | negative | neutral
Scores:  confidence probability for each label

Provides:
    1. Utterance-level sentiment
    2. Speaker-level aggregated sentiment
    3. Section-level sentiment (prepared_remarks vs Q&A)
    4. Quarter-over-quarter sentiment delta
    5. Composite NLP score per transcript
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SentimentScore:
    label: str          # positive | negative | neutral
    positive: float     # probability
    negative: float
    neutral: float
    compound: float     # positive - negative  ∈ [-1, 1]
    confidence: float   # max probability


@dataclass
class TranscriptSentiment:
    ticker: str
    quarter: str
    overall: SentimentScore
    ceo_sentiment: Optional[SentimentScore]
    cfo_sentiment: Optional[SentimentScore]
    analyst_sentiment: Optional[SentimentScore]
    prepared_sentiment: SentimentScore
    qa_sentiment: SentimentScore
    sentiment_shift: float      # prepared → QA compound delta
    utterance_scores: list      # list of (speaker, section, SentimentScore)


# ---------------------------------------------------------------------------
# FinBERT wrapper
# ---------------------------------------------------------------------------

class FinBERTScorer:
    """
    Wraps ProsusAI/finbert for earnings call sentiment analysis.

    Falls back to a lexicon-based scorer if transformers/GPU
    unavailable (useful for CI / testing without heavy deps).
    """

    FINBERT_MODEL = "ProsusAI/finbert"

    # Financial sentiment lexicon (fallback)
    POSITIVE_WORDS = {
        "strong", "growth", "record", "exceeded", "beat", "raised", "momentum",
        "confident", "optimistic", "opportunity", "accelerating", "robust",
        "outperform", "increase", "higher", "positive", "expand", "gain",
        "profit", "revenue", "margin", "improve", "excellent", "outstanding",
        "committed", "excited", "pleased", "healthy", "sustained", "innovative",
    }
    NEGATIVE_WORDS = {
        "decline", "loss", "miss", "below", "headwind", "challenge", "concern",
        "uncertain", "difficult", "weak", "pressure", "risk", "lower", "reduced",
        "disappointing", "slowdown", "cautious", "volatile", "uncertainty",
        "unfavorable", "deteriorate", "adverse", "impairment", "restructure",
    }
    HEDGE_WORDS = {
        "may", "might", "could", "possibly", "potentially", "expect", "anticipate",
        "approximately", "roughly", "around", "believe", "think", "assume",
        "subject to", "contingent", "dependent", "uncertain",
    }

    def __init__(self, use_transformers: bool = True):
        self.pipeline = None
        self.use_transformers = use_transformers
        if use_transformers:
            self._load_finbert()

    def _load_finbert(self):
        """Load FinBERT pipeline. Falls back to lexicon on failure."""
        try:
            from transformers import pipeline
            print("  Loading FinBERT (ProsusAI/finbert)...")
            self.pipeline = pipeline(
                "text-classification",
                model=self.FINBERT_MODEL,
                return_all_scores=True,
                truncation=True,
                max_length=512,
            )
            print("  FinBERT loaded successfully.")
        except Exception as e:
            print(f"  FinBERT unavailable ({e}). Using lexicon fallback.")
            self.pipeline = None

    def _lexicon_score(self, text: str) -> SentimentScore:
        """
        Fast lexicon-based fallback scorer.
        Uses financial word lists to estimate sentiment.
        """
        words = set(text.lower().split())
        pos_count = len(words & self.POSITIVE_WORDS)
        neg_count = len(words & self.NEGATIVE_WORDS)
        total = max(pos_count + neg_count, 1)

        pos_prob  = pos_count / (total + 2)   # smoothed
        neg_prob  = neg_count / (total + 2)
        neu_prob  = 1 - pos_prob - neg_prob
        neu_prob  = max(neu_prob, 0)

        # Renormalise
        s = pos_prob + neg_prob + neu_prob
        pos_prob /= s; neg_prob /= s; neu_prob /= s

        compound = float(pos_prob - neg_prob)

        if compound > 0.1:
            label = "positive"
            conf  = pos_prob
        elif compound < -0.1:
            label = "negative"
            conf  = neg_prob
        else:
            label = "neutral"
            conf  = neu_prob

        return SentimentScore(
            label=label,
            positive=round(pos_prob, 4),
            negative=round(neg_prob, 4),
            neutral=round(neu_prob, 4),
            compound=round(compound, 4),
            confidence=round(conf, 4),
        )

    def score_text(self, text: str) -> SentimentScore:
        """
        Score a text snippet. Uses FinBERT if available, else lexicon.
        Handles long texts by chunking and averaging.
        """
        if not text or len(text.strip()) < 5:
            return SentimentScore("neutral", 0.0, 0.0, 1.0, 0.0, 1.0)

        if self.pipeline is not None:
            return self._finbert_score(text)
        return self._lexicon_score(text)

    def _finbert_score(self, text: str) -> SentimentScore:
        """Score using FinBERT — chunk if text > 512 tokens."""
        # Split into ~400-word chunks for long texts
        words  = text.split()
        chunks = []
        chunk_size = 400
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))

        all_pos, all_neg, all_neu = [], [], []

        for chunk in chunks:
            try:
                result = self.pipeline(chunk)[0]
                scores = {r["label"].lower(): r["score"] for r in result}
                all_pos.append(scores.get("positive", 0))
                all_neg.append(scores.get("negative", 0))
                all_neu.append(scores.get("neutral",  0))
            except Exception:
                s = self._lexicon_score(chunk)
                all_pos.append(s.positive)
                all_neg.append(s.negative)
                all_neu.append(s.neutral)

        pos = float(np.mean(all_pos))
        neg = float(np.mean(all_neg))
        neu = float(np.mean(all_neu))
        compound = pos - neg

        if compound > 0.1:
            label, conf = "positive", pos
        elif compound < -0.1:
            label, conf = "negative", neg
        else:
            label, conf = "neutral", neu

        return SentimentScore(
            label=label,
            positive=round(pos, 4),
            negative=round(neg, 4),
            neutral=round(neu, 4),
            compound=round(compound, 4),
            confidence=round(conf, 4),
        )

    def _aggregate_scores(self, scores: list) -> SentimentScore:
        """Average multiple SentimentScores into one."""
        if not scores:
            return SentimentScore("neutral", 0.0, 0.0, 1.0, 0.0, 1.0)
        pos = np.mean([s.positive for s in scores])
        neg = np.mean([s.negative for s in scores])
        neu = np.mean([s.neutral  for s in scores])
        compound = float(pos - neg)
        if compound > 0.1:
            label, conf = "positive", float(pos)
        elif compound < -0.1:
            label, conf = "negative", float(neg)
        else:
            label, conf = "neutral",  float(neu)
        return SentimentScore(
            label=label,
            positive=round(float(pos), 4),
            negative=round(float(neg), 4),
            neutral=round(float(neu),  4),
            compound=round(compound,   4),
            confidence=round(conf,     4),
        )

    def score_transcript(self, transcript) -> TranscriptSentiment:
        """
        Score an entire Transcript object.

        Produces:
            - Per-utterance scores
            - Speaker-level aggregates (CEO, CFO, Analyst)
            - Section-level scores (prepared remarks vs Q&A)
            - Sentiment shift (prepared → QA)
            - Overall composite score
        """
        utterance_scores = []
        by_role  = {"CEO": [], "CFO": [], "Analyst": [], "prepared": [], "qa": []}

        for u in transcript.utterances:
            if u.speaker.role == "Operator" or u.word_count < 5:
                continue
            score = self.score_text(u.text)
            utterance_scores.append((u.speaker.name, u.speaker.role, u.section, score))

            if u.speaker.role in by_role:
                by_role[u.speaker.role].append(score)
            if u.section in by_role:
                by_role[u.section].append(score)

        overall   = self._aggregate_scores([s for _, _, _, s in utterance_scores])
        ceo_sent  = self._aggregate_scores(by_role["CEO"])   if by_role["CEO"]    else None
        cfo_sent  = self._aggregate_scores(by_role["CFO"])   if by_role["CFO"]    else None
        anlst_sent= self._aggregate_scores(by_role["Analyst"])if by_role["Analyst"]else None
        prep_sent = self._aggregate_scores(by_role["prepared"])
        qa_sent   = self._aggregate_scores(by_role["qa"])

        shift = round(qa_sent.compound - prep_sent.compound, 4)

        return TranscriptSentiment(
            ticker=transcript.ticker,
            quarter=transcript.quarter,
            overall=overall,
            ceo_sentiment=ceo_sent,
            cfo_sentiment=cfo_sent,
            analyst_sentiment=anlst_sent,
            prepared_sentiment=prep_sent,
            qa_sentiment=qa_sent,
            sentiment_shift=shift,
            utterance_scores=utterance_scores,
        )

    def score_corpus(self, transcripts: list) -> pd.DataFrame:
        """
        Score a list of Transcript objects and return a summary DataFrame.
        """
        rows = []
        for t in transcripts:
            ts = self.score_transcript(t)
            rows.append({
                "ticker":          ts.ticker,
                "quarter":         ts.quarter,
                "overall_compound":ts.overall.compound,
                "overall_label":   ts.overall.label,
                "ceo_compound":    ts.ceo_sentiment.compound if ts.ceo_sentiment else None,
                "cfo_compound":    ts.cfo_sentiment.compound if ts.cfo_sentiment else None,
                "analyst_compound":ts.analyst_sentiment.compound if ts.analyst_sentiment else None,
                "prep_compound":   ts.prepared_sentiment.compound,
                "qa_compound":     ts.qa_sentiment.compound,
                "sentiment_shift": ts.sentiment_shift,
            })
        return pd.DataFrame(rows)

    def compute_uncertainty(self, text: str) -> dict:
        """
        Measure uncertainty / hedge language in a text.
        Returns hedge word count, density, and uncertainty score.
        """
        words  = text.lower().split()
        n      = max(len(words), 1)
        hedges = [w for w in words if w in self.HEDGE_WORDS]
        density = len(hedges) / n
        score   = min(density * 10, 1.0)   # normalise to [0,1]
        return {
            "hedge_count":   len(hedges),
            "hedge_density": round(density, 4),
            "uncertainty_score": round(score, 4),
            "hedge_words":   hedges[:10],   # top 10 found
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from scraper.transcript_loader import load_synthetic_transcript

    scorer = FinBERTScorer(use_transformers=False)   # lexicon fallback for demo

    print("=" * 60)
    print("FinBERT Sentiment Scorer Demo  (lexicon fallback)")
    print("=" * 60)

    transcripts = [
        load_synthetic_transcript("AAPL"),
        load_synthetic_transcript("MSFT"),
    ]

    for t in transcripts:
        ts = scorer.score_transcript(t)
        print(f"\n[{ts.ticker}] {ts.quarter}")
        print(f"  Overall     : {ts.overall.label:>8}  compound={ts.overall.compound:+.4f}")
        if ts.ceo_sentiment:
            print(f"  CEO         : {ts.ceo_sentiment.label:>8}  compound={ts.ceo_sentiment.compound:+.4f}")
        if ts.cfo_sentiment:
            print(f"  CFO         : {ts.cfo_sentiment.label:>8}  compound={ts.cfo_sentiment.compound:+.4f}")
        if ts.analyst_sentiment:
            print(f"  Analysts    : {ts.analyst_sentiment.label:>8}  compound={ts.analyst_sentiment.compound:+.4f}")
        print(f"  Prepared    : {ts.prepared_sentiment.label:>8}  compound={ts.prepared_sentiment.compound:+.4f}")
        print(f"  Q&A         : {ts.qa_sentiment.label:>8}  compound={ts.qa_sentiment.compound:+.4f}")
        print(f"  Shift P→QA  : {ts.sentiment_shift:+.4f}")

        unc = scorer.compute_uncertainty(t.management_text)
        print(f"  Uncertainty : {unc['uncertainty_score']:.4f}  "
              f"(hedge density={unc['hedge_density']:.3f})")

    print("\nCorpus summary:")
    df = scorer.score_corpus(transcripts)
    print(df.to_string(index=False))
