# AI Earnings Call Analyst — NLP Pipeline

Scrapes earnings call transcripts, runs FinBERT sentiment scoring and BERTopic topic modeling, extracts forward guidance signals, and maps sentiment delta to next-day price returns. Produces alpha signals from management tone and analyst Q&A dynamics.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-36%20passing-brightgreen)](#testing)

---

## What this does

| Module | Description |
|--------|-------------|
| `scraper/transcript_loader.py` | Transcript loader, text cleaner, speaker diarization, section splitter, JSON storage |
| `nlp/finbert_scorer.py` | FinBERT sentiment scoring per utterance, section, and speaker |
| `nlp/topic_model.py` | BERTopic topic modeling across transcript corpus |
| `nlp/guidance_extractor.py` | Forward guidance detection + sentiment delta |
| `nlp/uncertainty_scorer.py` | Hedge word frequency, uncertainty language detector |
| `signals/alpha_gen.py` | Sentiment → next-day return signal generator |
| `backtest/event_study.py` | Event study: cumulative abnormal returns around earnings |
| `app/dashboard.py` | Streamlit app: ticker input → full transcript analysis |

---

## Methodology

### Transcript Pipeline

```
Raw HTML/Text → Clean → Speaker Diarization → Section Split → JSON
```

Sections detected: **Prepared Remarks** (management monologue) and **Q&A** (analyst questions + management responses).

### FinBERT Sentiment

FinBERT (fine-tuned BERT on financial text) scores each utterance as Positive / Negative / Neutral with a confidence score. Key signals:

- CEO tone vs CFO tone delta
- Prepared remarks vs Q&A sentiment shift
- Quarter-over-quarter sentiment change

### Forward Guidance Extraction

Rule-based NER + keyword matching identifies forward-looking statements:
```
"We expect", "We anticipate", "Going forward", "In Q4 we will..."
```
Guidance sentiment delta (vs last quarter) is the primary alpha signal.

### Alpha Signal

```
signal = sentiment_surprise × guidance_delta × uncertainty_inverse
```

Long if signal > threshold, short if signal < -threshold, flat otherwise.

---

## Results

| Metric | Value |
|--------|-------|
| Signal hit rate | TBD |
| Avg return per signal | TBD |
| Sharpe (earnings day strategy) | TBD |
| FinBERT accuracy vs baseline | TBD |

---

## Output Samples

![Day 1 — Transcript Pipeline](outputs/day1_transcript_pipeline.png)

---

## Project Structure

```
earnings-call-analyst/
├── scraper/
│   └── transcript_loader.py   # Loader, cleaner, diarizer, JSON storage
├── nlp/
│   ├── finbert_scorer.py      # FinBERT sentiment pipeline
│   ├── topic_model.py         # BERTopic topic modeling
│   ├── guidance_extractor.py  # Forward guidance signals
│   └── uncertainty_scorer.py  # Hedge word analysis
├── signals/
│   └── alpha_gen.py           # Sentiment → alpha signal
├── backtest/
│   └── event_study.py         # CAR event study
├── app/
│   └── dashboard.py           # Streamlit dashboard
├── tests/
│   └── test_transcript_loader.py
├── outputs/
├── data/
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/yuvrajsingh1097/earnings-call-analyst
cd earnings-call-analyst
pip install -r requirements.txt
```

```bash
python scraper/transcript_loader.py    # pipeline demo
python -m pytest tests/ -v             # run tests
streamlit run app/dashboard.py         # launch dashboard
```

---

## Testing

```
36 passed in 0.09s
```

---

## License

MIT
