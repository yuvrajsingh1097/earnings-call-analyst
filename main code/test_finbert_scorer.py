"""
Unit Tests — FinBERT Sentiment Scorer
======================================
Run with: python -m pytest tests/test_finbert_scorer.py -v
"""

import pytest
import pandas as pd
from nlp.finbert_scorer import FinBERTScorer, SentimentScore, TranscriptSentiment


@pytest.fixture(scope="module")
def scorer():
    return FinBERTScorer(use_transformers=False)


@pytest.fixture(scope="module")
def transcripts():
    import sys; sys.path.insert(0, ".")
    from scraper.transcript_loader import load_synthetic_transcript
    return [load_synthetic_transcript("AAPL"), load_synthetic_transcript("MSFT")]


# ---------------------------------------------------------------------------
# 1. Lexicon scorer — basic
# ---------------------------------------------------------------------------

class TestLexiconScorer:

    def test_positive_text(self, scorer):
        s = scorer.score_text("Revenue grew strongly, record profits, excellent growth momentum")
        assert s.label == "positive"
        assert s.compound > 0

    def test_negative_text(self, scorer):
        s = scorer.score_text("Revenue declined, loss widened, weak demand, significant headwinds")
        assert s.label == "negative"
        assert s.compound < 0

    def test_neutral_text(self, scorer):
        s = scorer.score_text("The company reported results for the quarter")
        assert s.label == "neutral"

    def test_empty_text(self, scorer):
        s = scorer.score_text("")
        assert s.label == "neutral"

    def test_short_text(self, scorer):
        s = scorer.score_text("hi")
        assert isinstance(s, SentimentScore)

    def test_probabilities_sum_to_one(self, scorer):
        s = scorer.score_text("Strong revenue growth exceeded expectations this quarter")
        total = s.positive + s.negative + s.neutral
        assert abs(total - 1.0) < 0.01

    def test_compound_in_range(self, scorer):
        s = scorer.score_text("Strong growth with some challenges ahead")
        assert -1.0 <= s.compound <= 1.0

    def test_confidence_positive(self, scorer):
        s = scorer.score_text("Record revenue and profit growth")
        assert s.confidence > 0

    def test_returns_sentiment_score(self, scorer):
        assert isinstance(scorer.score_text("test"), SentimentScore)


# ---------------------------------------------------------------------------
# 2. Transcript scoring
# ---------------------------------------------------------------------------

class TestTranscriptScoring:

    def test_returns_transcript_sentiment(self, scorer, transcripts):
        ts = scorer.score_transcript(transcripts[0])
        assert isinstance(ts, TranscriptSentiment)

    def test_ticker_preserved(self, scorer, transcripts):
        ts = scorer.score_transcript(transcripts[0])
        assert ts.ticker == "AAPL"

    def test_overall_sentiment_valid(self, scorer, transcripts):
        ts = scorer.score_transcript(transcripts[0])
        assert ts.overall.label in {"positive", "negative", "neutral"}

    def test_ceo_sentiment_present(self, scorer, transcripts):
        ts = scorer.score_transcript(transcripts[0])
        assert ts.ceo_sentiment is not None

    def test_cfo_sentiment_present(self, scorer, transcripts):
        ts = scorer.score_transcript(transcripts[0])
        assert ts.cfo_sentiment is not None

    def test_analyst_sentiment_present(self, scorer, transcripts):
        ts = scorer.score_transcript(transcripts[0])
        assert ts.analyst_sentiment is not None

    def test_prepared_sentiment_present(self, scorer, transcripts):
        ts = scorer.score_transcript(transcripts[0])
        assert isinstance(ts.prepared_sentiment, SentimentScore)

    def test_qa_sentiment_present(self, scorer, transcripts):
        ts = scorer.score_transcript(transcripts[0])
        assert isinstance(ts.qa_sentiment, SentimentScore)

    def test_sentiment_shift_is_float(self, scorer, transcripts):
        ts = scorer.score_transcript(transcripts[0])
        assert isinstance(ts.sentiment_shift, float)

    def test_utterance_scores_populated(self, scorer, transcripts):
        ts = scorer.score_transcript(transcripts[0])
        assert len(ts.utterance_scores) > 0

    def test_utterance_score_structure(self, scorer, transcripts):
        ts = scorer.score_transcript(transcripts[0])
        name, role, section, score = ts.utterance_scores[0]
        assert isinstance(name, str)
        assert isinstance(score, SentimentScore)

    def test_msft_scores(self, scorer, transcripts):
        ts = scorer.score_transcript(transcripts[1])
        assert ts.ticker == "MSFT"
        assert ts.overall.label in {"positive", "negative", "neutral"}


# ---------------------------------------------------------------------------
# 3. Corpus scoring
# ---------------------------------------------------------------------------

class TestCorpusScoring:

    def test_returns_dataframe(self, scorer, transcripts):
        df = scorer.score_corpus(transcripts)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, scorer, transcripts):
        df = scorer.score_corpus(transcripts)
        assert len(df) == len(transcripts)

    def test_columns_present(self, scorer, transcripts):
        df = scorer.score_corpus(transcripts)
        for col in ["ticker", "overall_compound", "ceo_compound", "sentiment_shift"]:
            assert col in df.columns

    def test_compounds_in_range(self, scorer, transcripts):
        df = scorer.score_corpus(transcripts)
        assert (df["overall_compound"].between(-1, 1)).all()

    def test_tickers_correct(self, scorer, transcripts):
        df = scorer.score_corpus(transcripts)
        assert set(df["ticker"]) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# 4. Uncertainty scorer
# ---------------------------------------------------------------------------

class TestUncertainty:

    def test_returns_dict(self, scorer):
        u = scorer.compute_uncertainty("We expect revenue may potentially grow")
        assert isinstance(u, dict)

    def test_keys_present(self, scorer):
        u = scorer.compute_uncertainty("We expect growth")
        for k in ["hedge_count", "hedge_density", "uncertainty_score", "hedge_words"]:
            assert k in u

    def test_high_uncertainty_text(self, scorer):
        text = "We may possibly expect approximately potentially uncertain results"
        u = scorer.compute_uncertainty(text)
        assert u["hedge_count"] >= 3

    def test_low_uncertainty_text(self, scorer):
        text = "Revenue was exactly 100 billion dollars"
        u = scorer.compute_uncertainty(text)
        assert u["hedge_count"] == 0

    def test_score_in_range(self, scorer):
        u = scorer.compute_uncertainty("We believe results may potentially improve")
        assert 0 <= u["uncertainty_score"] <= 1

    def test_empty_text(self, scorer):
        u = scorer.compute_uncertainty("")
        assert u["hedge_count"] == 0
