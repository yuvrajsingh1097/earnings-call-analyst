"""
Unit Tests — Transcript Scraper & Cleaner
==========================================
Run with: python -m pytest tests/test_transcript_loader.py -v
"""

import pytest
import json
import os
from scraper.transcript_loader import (
    clean_text, classify_speaker, is_management, detect_section,
    load_synthetic_transcript, save_transcript,
    Transcript, Utterance, Speaker,
)


# ---------------------------------------------------------------------------
# 1. Text cleaning
# ---------------------------------------------------------------------------

class TestCleanText:

    def test_removes_html_tags(self):
        assert "<p>" not in clean_text("<p>Hello world</p>")

    def test_removes_filler_words(self):
        result = clean_text("So um basically we are, you know, growing")
        assert "um" not in result.lower()

    def test_normalises_whitespace(self):
        result = clean_text("hello   world\n\n\nfoo")
        assert "   " not in result
        assert "\n\n\n" not in result

    def test_normalises_percentage(self):
        result = clean_text("Revenue grew 15%")
        assert "percent" in result

    def test_strips_html_entities(self):
        result = clean_text("AT&amp;T reported &nbsp;results")
        assert "&amp;" not in result
        assert "&nbsp;" not in result

    def test_preserves_content(self):
        text = "Revenue increased significantly year over year"
        result = clean_text(text)
        assert "Revenue" in result
        assert "year over year" in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_returns_string(self):
        assert isinstance(clean_text("hello"), str)


# ---------------------------------------------------------------------------
# 2. Speaker classification
# ---------------------------------------------------------------------------

class TestClassifySpeaker:

    def test_ceo_classification(self):
        assert classify_speaker("Tim Cook", "Chief Executive Officer") == "CEO"

    def test_cfo_classification(self):
        assert classify_speaker("Luca Maestri", "Chief Financial Officer") == "CFO"

    def test_operator_classification(self):
        assert classify_speaker("Operator", "Operator") == "Operator"

    def test_analyst_from_title(self):
        role = classify_speaker("John Smith", "Analyst at Goldman Sachs")
        assert role == "Analyst"

    def test_analyst_from_firm(self):
        role = classify_speaker("Jane Doe", "Morgan Stanley")
        assert role == "Analyst"

    def test_unknown_default(self):
        role = classify_speaker("Random Person", "")
        assert role == "Unknown"

    def test_management_flag_ceo(self):
        assert is_management("CEO") is True

    def test_management_flag_analyst(self):
        assert is_management("Analyst") is False

    def test_management_flag_operator(self):
        assert is_management("Operator") is False


# ---------------------------------------------------------------------------
# 3. Section detection
# ---------------------------------------------------------------------------

class TestSectionDetection:

    def test_qa_trigger_detected(self):
        text = "We will now begin the question-and-answer session"
        assert detect_section(text, "prepared_remarks") == "qa"

    def test_prepared_remarks_unchanged(self):
        text = "Revenue grew 15 percent year over year"
        assert detect_section(text, "prepared_remarks") == "prepared_remarks"

    def test_qa_section_persists(self):
        text = "Thank you for your question"
        assert detect_section(text, "qa") == "qa"


# ---------------------------------------------------------------------------
# 4. Synthetic transcript loading
# ---------------------------------------------------------------------------

class TestSyntheticTranscript:

    @pytest.fixture
    def aapl(self):
        return load_synthetic_transcript("AAPL")

    @pytest.fixture
    def msft(self):
        return load_synthetic_transcript("MSFT")

    def test_returns_transcript(self, aapl):
        assert isinstance(aapl, Transcript)

    def test_ticker_correct(self, aapl):
        assert aapl.ticker == "AAPL"

    def test_has_utterances(self, aapl):
        assert len(aapl.utterances) > 0

    def test_management_text_nonempty(self, aapl):
        assert len(aapl.management_text) > 0

    def test_analyst_text_nonempty(self, aapl):
        assert len(aapl.analyst_text) > 0

    def test_prepared_remarks_nonempty(self, aapl):
        assert len(aapl.prepared_remarks) > 0

    def test_qa_section_nonempty(self, aapl):
        assert len(aapl.qa_section) > 0

    def test_total_words_positive(self, aapl):
        assert aapl.total_words > 0

    def test_unknown_ticker_returns_none(self):
        assert load_synthetic_transcript("ZZZZ") is None

    def test_utterance_word_count(self, aapl):
        for u in aapl.utterances:
            assert u.word_count == len(u.text.split())

    def test_msft_loads(self, msft):
        assert msft.ticker == "MSFT"
        assert len(msft.utterances) > 0

    def test_sections_assigned(self, aapl):
        sections = {u.section for u in aapl.utterances}
        assert "prepared_remarks" in sections
        assert "qa" in sections


# ---------------------------------------------------------------------------
# 5. JSON storage
# ---------------------------------------------------------------------------

class TestJSONStorage:

    def test_saves_file(self, tmp_path):
        t = load_synthetic_transcript("AAPL")
        fpath = save_transcript(t, str(tmp_path))
        assert os.path.exists(fpath)

    def test_valid_json(self, tmp_path):
        t = load_synthetic_transcript("AAPL")
        fpath = save_transcript(t, str(tmp_path))
        with open(fpath) as f:
            data = json.load(f)
        assert "ticker" in data
        assert "utterances" in data
        assert "stats" in data

    def test_stats_correct(self, tmp_path):
        t = load_synthetic_transcript("AAPL")
        fpath = save_transcript(t, str(tmp_path))
        with open(fpath) as f:
            data = json.load(f)
        assert data["stats"]["total_words"] == t.total_words
        assert data["stats"]["n_utterances"] == len(t.utterances)

    def test_utterances_serialised(self, tmp_path):
        t = load_synthetic_transcript("MSFT")
        fpath = save_transcript(t, str(tmp_path))
        with open(fpath) as f:
            data = json.load(f)
        assert len(data["utterances"]) == len(t.utterances)
        assert "text" in data["utterances"][0]
        assert "speaker_role" in data["utterances"][0]
