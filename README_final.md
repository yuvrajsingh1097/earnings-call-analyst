# AI Earnings Call Analyst — NLP Pipeline

Scrapes earnings call transcripts, runs FinBERT sentiment scoring and LDA/BERTopic topic modeling, extracts forward guidance signals, and maps sentiment delta to next-day price returns. Produces alpha signals from management tone and analyst Q&A dynamics.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-193%20passing-brightgreen)](#testing)

---

## What this does

| Module | Description |
|--------|-------------|
| `scraper/transcript_loader.py` | Transcript loader, text cleaner, speaker diarization, section splitter, JSON storage |
| `nlp/finbert_scorer.py` | FinBERT sentiment per utterance, section, speaker; uncertainty scorer |
| `nlp/topic_model.py` | LDA + BERTopic topic modeling across transcript corpus |
| `nlp/guidance_extractor.py` | Forward guidance detection, category classifier, surprise score |
| `signals/alpha_gen.py` | Price data loader, event study (CAR), sentiment → return mapping |
| `backtest/engine.py` | Full backtest engine with Sharpe/Sortino/MDD, sector breakdown, monthly heatmap |
| `app/dashboard.py` | Streamlit interactive dashboard |

---

## Methodology

### Transcript Pipeline
```
Raw HTML/Text → Clean → Speaker Diarization → Section Split → JSON
```

### FinBERT Sentiment
FinBERT (ProsusAI/finbert) scores each utterance as Positive / Negative / Neutral.
Key signals: CEO tone vs CFO tone, prepared remarks vs Q&A shift, uncertainty hedge density.

### Forward Guidance Extraction
Rule-based NER + modal/tense analysis detects forward-looking statements.
Categories: revenue, margin, capex, volume, macro, competition.

### Alpha Signal
```
signal = sentiment_surprise × guidance_delta × (1 - uncertainty)
```

### Backtest
Event-driven backtest: long/short on earnings date, 5-day hold, 10bps transaction cost.

---

## Results

| Metric | Value |
|--------|-------|
| Backtest win rate | 52% |
| Best sector (Energy) avg PnL | +0.65% |
| Signal hit rate | 50–60% |
| Tests passing | 193 |

---

## Output Samples

![Dashboard Preview](outputs/dashboard_preview.png)
![Backtest Results](outputs/backtest_results.png)
![Sentiment Analysis](outputs/sentiment_analysis.png)
![Topic Modeling](outputs/topic_modeling.png)
![Guidance Extraction](outputs/guidance_extraction.png)
![Sentiment Return Mapping](outputs/sentiment_return_mapping.png)

---

## Project Structure

```
earnings-call-analyst/
├── scraper/transcript_loader.py
├── nlp/
│   ├── finbert_scorer.py
│   ├── topic_model.py
│   └── guidance_extractor.py
├── signals/alpha_gen.py
├── backtest/engine.py
├── app/dashboard.py
├── tests/  (193 tests)
└── outputs/
```

---

## Installation

```bash
git clone https://github.com/yuvrajsingh1097/earnings-call-analyst
cd earnings-call-analyst
pip install -r requirements.txt
streamlit run app/dashboard.py
```

---

## License
MIT
