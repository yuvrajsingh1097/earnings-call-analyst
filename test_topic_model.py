"""
Unit Tests — Topic Modeling
============================
Run with: python -m pytest tests/test_topic_model.py -v
"""

import pytest
import pandas as pd
from nlp.topic_model import (
    preprocess_for_topics, build_corpus, LDATopicModel,
    BERTopicModel, topic_distribution_per_ticker,
    top_keywords_per_section, keyword_frequency_tracker,
    Topic, TopicModelResult,
)


@pytest.fixture(scope="module")
def transcripts():
    import sys; sys.path.insert(0, ".")
    from scraper.transcript_loader import load_synthetic_transcript
    return [load_synthetic_transcript("AAPL"), load_synthetic_transcript("MSFT")]


@pytest.fixture(scope="module")
def docs(transcripts):
    return build_corpus(transcripts)


@pytest.fixture(scope="module")
def lda_result(docs):
    return LDATopicModel(n_topics=3).fit(docs)


# ---------------------------------------------------------------------------
# 1. Preprocessor
# ---------------------------------------------------------------------------

class TestPreprocess:

    def test_removes_stopwords(self):
        tokens = preprocess_for_topics("the company delivered strong momentum")
        assert "the" not in tokens
        assert "strong" in tokens or "momentum" in tokens

    def test_min_length_filter(self):
        tokens = preprocess_for_topics("we do it a go")
        assert all(len(t) >= 3 for t in tokens)

    def test_lowercased(self):
        tokens = preprocess_for_topics("Revenue GREW strongly")
        assert all(t == t.lower() for t in tokens)

    def test_returns_list(self):
        assert isinstance(preprocess_for_topics("hello world"), list)

    def test_empty_text(self):
        assert preprocess_for_topics("") == []


# ---------------------------------------------------------------------------
# 2. Corpus builder
# ---------------------------------------------------------------------------

class TestBuildCorpus:

    def test_returns_list(self, docs):
        assert isinstance(docs, list)

    def test_nonempty(self, docs):
        assert len(docs) > 0

    def test_doc_keys(self, docs):
        for k in ["ticker", "quarter", "role", "section", "text", "tokens"]:
            assert k in docs[0]

    def test_no_operator_docs(self, docs):
        assert all(d["role"] != "Operator" for d in docs)

    def test_section_filter(self, transcripts):
        qa_docs = build_corpus(transcripts, section="qa")
        assert all(d["section"] == "qa" for d in qa_docs)

    def test_prepared_filter(self, transcripts):
        prep_docs = build_corpus(transcripts, section="prepared_remarks")
        assert all(d["section"] == "prepared_remarks" for d in prep_docs)

    def test_tokens_are_lists(self, docs):
        assert all(isinstance(d["tokens"], list) for d in docs)


# ---------------------------------------------------------------------------
# 3. LDA
# ---------------------------------------------------------------------------

class TestLDA:

    def test_returns_result(self, lda_result):
        assert isinstance(lda_result, TopicModelResult)

    def test_model_type(self, lda_result):
        assert lda_result.model_type == "lda"

    def test_n_topics(self, lda_result):
        assert lda_result.n_topics <= 3

    def test_topics_list(self, lda_result):
        assert isinstance(lda_result.topics, list)
        assert len(lda_result.topics) > 0

    def test_topic_structure(self, lda_result):
        t = lda_result.topics[0]
        assert isinstance(t, Topic)
        assert isinstance(t.keywords, list)
        assert len(t.keywords) > 0

    def test_doc_topics_length(self, lda_result, docs):
        assert len(lda_result.doc_topics) == lda_result.n_docs

    def test_keywords_are_tuples(self, lda_result):
        for topic in lda_result.topics:
            for kw in topic.keywords:
                assert len(kw) == 2

    def test_doc_count_sums(self, lda_result):
        total = sum(t.doc_count for t in lda_result.topics)
        assert total == lda_result.n_docs


# ---------------------------------------------------------------------------
# 4. BERTopic (fallback)
# ---------------------------------------------------------------------------

class TestBERTopic:

    def test_returns_result(self, docs):
        result = BERTopicModel(n_topics=3).fit(docs)
        assert isinstance(result, TopicModelResult)

    def test_has_topics(self, docs):
        result = BERTopicModel(n_topics=3).fit(docs)
        assert result.n_topics > 0

    def test_doc_topics_populated(self, docs):
        result = BERTopicModel(n_topics=3).fit(docs)
        assert len(result.doc_topics) > 0


# ---------------------------------------------------------------------------
# 5. Utilities
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_keyword_tracker_returns_df(self, transcripts):
        df = keyword_frequency_tracker(transcripts, ["revenue", "growth"])
        assert isinstance(df, pd.DataFrame)

    def test_keyword_tracker_columns(self, transcripts):
        df = keyword_frequency_tracker(transcripts, ["revenue", "growth"])
        assert "ticker" in df.columns
        assert "revenue" in df.columns

    def test_keyword_tracker_rows(self, transcripts):
        df = keyword_frequency_tracker(transcripts, ["revenue"])
        assert len(df) == len(transcripts)

    def test_section_keywords_returns_dict(self, docs):
        result = top_keywords_per_section(docs)
        assert isinstance(result, dict)

    def test_section_keywords_has_sections(self, docs):
        result = top_keywords_per_section(docs)
        assert "prepared_remarks" in result
        assert "qa" in result

    def test_section_keywords_are_tuples(self, docs):
        result = top_keywords_per_section(docs)
        for section, kws in result.items():
            for kw in kws:
                assert len(kw) == 2

    def test_topic_distribution_df(self, lda_result, docs):
        df = topic_distribution_per_ticker(docs, lda_result.doc_topics, ["AAPL", "MSFT"])
        assert isinstance(df, pd.DataFrame)
        assert "ticker" in df.columns
