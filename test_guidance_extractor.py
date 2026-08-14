"""
Unit Tests — Forward Guidance Extractor
=========================================
Run with: python -m pytest tests/test_guidance_extractor.py -v
"""

import pytest
import pandas as pd
from nlp.guidance_extractor import (
    GuidanceExtractor, GuidanceReport, GuidanceStatement,
    compute_sentiment_delta, guidance_surprise_score,
    guidance_corpus_summary, GUIDANCE_RE,
)


@pytest.fixture(scope="module")
def transcripts():
    import sys; sys.path.insert(0, ".")
    from scraper.transcript_loader import load_synthetic_transcript
    return [load_synthetic_transcript("AAPL"), load_synthetic_transcript("MSFT")]


@pytest.fixture(scope="module")
def extractor():
    return GuidanceExtractor()


@pytest.fixture(scope="module")
def reports(transcripts, extractor):
    return [extractor.extract(t) for t in transcripts]


# ---------------------------------------------------------------------------
# 1. Pattern matching
# ---------------------------------------------------------------------------

class TestGuidancePatterns:

    def test_detects_we_expect(self):
        assert GUIDANCE_RE.search("We expect revenue to grow next quarter")

    def test_detects_going_forward(self):
        assert GUIDANCE_RE.search("Going forward we will invest in AI")

    def test_detects_guidance_phrase(self):
        assert GUIDANCE_RE.search("Our guidance for Q4 is strong growth")

    def test_no_match_plain_text(self):
        assert not GUIDANCE_RE.search("Revenue was 85 billion last quarter")

    def test_detects_we_anticipate(self):
        assert GUIDANCE_RE.search("We anticipate margins will expand")


# ---------------------------------------------------------------------------
# 2. Sentence extractor
# ---------------------------------------------------------------------------

class TestSentenceExtractor:

    def test_extracts_guidance_sentences(self, extractor):
        text = ("Revenue was strong last quarter. "
                "We expect Q4 growth to accelerate. "
                "We are confident in our trajectory.")
        sents = extractor.extract_guidance_sentences(text)
        assert len(sents) >= 1
        assert any("expect" in s.lower() for s in sents)

    def test_skips_short_sentences(self, extractor):
        sents = extractor.extract_guidance_sentences("We expect. Short.")
        # "We expect." has only 2 words → should be skipped
        assert all(len(s.split()) >= 5 for s in sents)

    def test_returns_list(self, extractor):
        result = extractor.extract_guidance_sentences("We expect growth to continue strongly.")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 3. Category classifier
# ---------------------------------------------------------------------------

class TestCategoryClassifier:

    def test_revenue_category(self, extractor):
        assert extractor.classify_category("We expect revenue to grow 10 percent") == "revenue"

    def test_margin_category(self, extractor):
        assert extractor.classify_category("Gross margin expected at 46 percent") == "margin"

    def test_capex_category(self, extractor):
        assert extractor.classify_category("Capital expenditure will increase next year") == "capex"

    def test_macro_category(self, extractor):
        assert extractor.classify_category("FX headwinds from currency fluctuations") == "macro"

    def test_general_fallback(self, extractor):
        cat = extractor.classify_category("We plan to continue our efforts going forward")
        assert isinstance(cat, str)


# ---------------------------------------------------------------------------
# 4. Sentiment scorer
# ---------------------------------------------------------------------------

class TestSentimentScorer:

    def test_positive_guidance(self, extractor):
        label, score = extractor.score_sentiment(
            "We expect strong growth and robust expansion ahead"
        )
        assert label == "positive"
        assert score > 0

    def test_negative_guidance(self, extractor):
        label, score = extractor.score_sentiment(
            "We face headwinds and pressure from weak demand and decline"
        )
        assert label == "negative"
        assert score < 0

    def test_score_in_range(self, extractor):
        _, score = extractor.score_sentiment("We expect moderate growth next quarter")
        assert -1.0 <= score <= 1.0

    def test_returns_tuple(self, extractor):
        result = extractor.score_sentiment("Going forward we anticipate improvement")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 5. Quantitative detection
# ---------------------------------------------------------------------------

class TestQuantitativeDetection:

    def test_detects_percentage(self, extractor):
        assert extractor.is_quantitative("Revenue expected to grow 15 percent") is True

    def test_detects_dollar(self, extractor):
        assert extractor.is_quantitative("We guide to $64 billion in revenue") is True

    def test_detects_range(self, extractor):
        assert extractor.is_quantitative("Growth expected between 28 and 30 percent") is True

    def test_no_numbers(self, extractor):
        assert extractor.is_quantitative("We expect strong growth going forward") is False


# ---------------------------------------------------------------------------
# 6. Full extraction
# ---------------------------------------------------------------------------

class TestFullExtraction:

    def test_returns_guidance_report(self, reports):
        assert isinstance(reports[0], GuidanceReport)

    def test_has_statements(self, reports):
        assert len(reports[0].statements) > 0

    def test_statement_structure(self, reports):
        s = reports[0].statements[0]
        assert isinstance(s, GuidanceStatement)
        assert isinstance(s.text, str)
        assert isinstance(s.sentiment_score, float)

    def test_overall_sentiment_in_range(self, reports):
        for r in reports:
            assert -1.0 <= r.overall_sentiment <= 1.0

    def test_counts_add_up(self, reports):
        for r in reports:
            assert r.n_quantitative + r.n_qualitative == len(r.statements)

    def test_ticker_preserved(self, reports, transcripts):
        for r, t in zip(reports, transcripts):
            assert r.ticker == t.ticker

    def test_only_management_statements(self, reports):
        for r in reports:
            for s in r.statements:
                assert s.speaker_role != "Analyst"


# ---------------------------------------------------------------------------
# 7. Sentiment delta + surprise
# ---------------------------------------------------------------------------

class TestSentimentDelta:

    def test_returns_dataframe(self, reports):
        df = compute_sentiment_delta(reports, "AAPL")
        assert isinstance(df, pd.DataFrame)

    def test_empty_for_unknown_ticker(self, reports):
        df = compute_sentiment_delta(reports, "ZZZZ")
        assert len(df) == 0

    def test_delta_is_none_for_first_quarter(self, reports):
        df = compute_sentiment_delta(reports, "AAPL")
        if len(df) > 0:
            assert df.iloc[0]["delta"] is None or pd.isna(df.iloc[0]["delta"])

    def test_surprise_score_in_range(self):
        score = guidance_surprise_score(0.5, 0.2)
        assert -1.0 <= score <= 1.0

    def test_positive_surprise(self):
        score = guidance_surprise_score(0.8, 0.2)
        assert score > 0

    def test_negative_surprise(self):
        score = guidance_surprise_score(0.1, 0.7)
        assert score < 0


# ---------------------------------------------------------------------------
# 8. Corpus summary
# ---------------------------------------------------------------------------

class TestCorpusSummary:

    def test_returns_dataframe(self, reports):
        df = guidance_corpus_summary(reports)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, reports):
        df = guidance_corpus_summary(reports)
        assert len(df) == len(reports)

    def test_columns_present(self, reports):
        df = guidance_corpus_summary(reports)
        for col in ["ticker", "quarter", "n_statements", "overall_sentiment"]:
            assert col in df.columns
