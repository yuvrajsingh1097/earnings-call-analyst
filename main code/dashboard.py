"""
Streamlit Dashboard — AI Earnings Call Analyst
================================================
Interactive dashboard that ties together all modules:
    - Transcript loading and display
    - FinBERT sentiment analysis
    - Topic modeling
    - Forward guidance extraction
    - Signal backtest results

Run with: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scraper.transcript_loader import load_synthetic_transcript, SYNTHETIC_TRANSCRIPTS
from nlp.finbert_scorer import FinBERTScorer
from nlp.topic_model import LDATopicModel, build_corpus, top_keywords_per_section
from nlp.guidance_extractor import GuidanceExtractor, guidance_surprise_score
from signals.alpha_gen import (
    synthetic_prices, get_market_returns,
    EarningsEvent, EventStudy, backtest_sentiment_signal,
)
from backtest.engine import (
    synthetic_universe, synthetic_signals,
    BacktestEngine, SECTORS,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Earnings Call Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: 600; }
    .metric-label { font-size: 12px; color: #8b949e; margin-top: 4px; }
    .positive { color: #3fb950; }
    .negative { color: #ff7b72; }
    .neutral  { color: #8b949e; }
    h1, h2, h3 { color: #e6edf3; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_scorer():
    return FinBERTScorer(use_transformers=False)


@st.cache_resource
def get_extractor():
    return GuidanceExtractor()


@st.cache_data
def load_transcript(ticker):
    return load_synthetic_transcript(ticker)


@st.cache_data
def run_full_analysis(ticker):
    scorer    = get_scorer()
    extractor = get_extractor()
    t  = load_transcript(ticker)
    ts = scorer.score_transcript(t)
    report = extractor.extract(t)
    docs   = build_corpus([t])
    lda    = LDATopicModel(n_topics=3)
    topic_result = lda.fit(docs)
    section_kws  = top_keywords_per_section(docs, top_n=8)
    surprise = guidance_surprise_score(
        ts.prepared_sentiment.compound,
        ts.analyst_sentiment.compound if ts.analyst_sentiment else 0,
    )
    return t, ts, report, topic_result, section_kws, surprise


@st.cache_data
def run_backtest():
    tickers  = list(SECTORS.keys())
    universe = synthetic_universe(tickers, n_days=504)
    signals  = synthetic_signals(tickers, universe, n_events_per_ticker=4)
    engine   = BacktestEngine(threshold=0.15, hold_days=5)
    return engine.run(signals, universe)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📊 Earnings Analyst")
st.sidebar.markdown("---")

available_tickers = list(SYNTHETIC_TRANSCRIPTS.keys())
ticker = st.sidebar.selectbox("Select Ticker", available_tickers, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("", [
    "📋 Transcript Overview",
    "🧠 Sentiment Analysis",
    "📚 Topic Modeling",
    "🔮 Forward Guidance",
    "📈 Signal & Backtest",
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Stack:**
- FinBERT (ProsusAI)
- LDA / BERTopic
- Rule-based NER
- Custom backtest engine
""")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with st.spinner(f"Analysing {ticker}..."):
    t, ts, report, topic_result, section_kws, surprise = run_full_analysis(ticker)


# ---------------------------------------------------------------------------
# Page 1 — Transcript Overview
# ---------------------------------------------------------------------------

if page == "📋 Transcript Overview":
    st.title(f"📋 {t.company_name} — {t.quarter}")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Words", f"{t.total_words:,}")
    with col2:
        st.metric("Management Words", f"{len(t.management_text.split()):,}")
    with col3:
        st.metric("Analyst Words", f"{len(t.analyst_text.split()):,}")
    with col4:
        st.metric("Utterances", len(t.utterances))

    st.markdown("---")

    # Speaker breakdown
    st.subheader("Speaker Breakdown")
    speaker_data = {}
    for u in t.utterances:
        key = f"{u.speaker.name} ({u.speaker.role})"
        speaker_data[key] = speaker_data.get(key, 0) + u.word_count

    df_speakers = pd.DataFrame(
        list(speaker_data.items()), columns=["Speaker", "Words"]
    ).sort_values("Words", ascending=False)
    st.bar_chart(df_speakers.set_index("Speaker"))

    st.markdown("---")

    # Transcript viewer
    st.subheader("Transcript")
    section_filter = st.radio("Section", ["All", "Prepared Remarks", "Q&A"], horizontal=True)

    for u in t.utterances:
        if section_filter == "Prepared Remarks" and u.section != "prepared_remarks":
            continue
        if section_filter == "Q&A" and u.section != "qa":
            continue

        role_color = {
            "CEO": "🔵", "CFO": "🟢", "Analyst": "🟠", "Operator": "⚫"
        }.get(u.speaker.role, "⚪")

        with st.expander(f"{role_color} {u.speaker.name} ({u.speaker.role}) — {u.word_count} words"):
            st.write(u.text)


# ---------------------------------------------------------------------------
# Page 2 — Sentiment Analysis
# ---------------------------------------------------------------------------

elif page == "🧠 Sentiment Analysis":
    st.title(f"🧠 Sentiment Analysis — {ticker}")

    # Overall sentiment
    overall = ts.overall
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        color = "positive" if overall.compound > 0.1 else "negative" if overall.compound < -0.1 else "neutral"
        st.metric("Overall Sentiment", overall.label.upper(),
                  delta=f"{overall.compound:+.3f}")
    with col2:
        st.metric("Positive Prob", f"{overall.positive:.1%}")
    with col3:
        st.metric("Negative Prob", f"{overall.negative:.1%}")
    with col4:
        st.metric("Sentiment Shift (P→QA)", f"{ts.sentiment_shift:+.3f}")

    st.markdown("---")

    # Speaker comparison
    st.subheader("Sentiment by Speaker Role")
    roles = []
    if ts.ceo_sentiment:
        roles.append(("CEO", ts.ceo_sentiment.compound))
    if ts.cfo_sentiment:
        roles.append(("CFO", ts.cfo_sentiment.compound))
    if ts.analyst_sentiment:
        roles.append(("Analysts", ts.analyst_sentiment.compound))
    roles.append(("Prepared Remarks", ts.prepared_sentiment.compound))
    roles.append(("Q&A Section", ts.qa_sentiment.compound))

    df_roles = pd.DataFrame(roles, columns=["Role", "Compound Score"])
    st.bar_chart(df_roles.set_index("Role"))

    st.markdown("---")

    # Utterance timeline
    st.subheader("Utterance-Level Sentiment Timeline")
    utt_data = []
    for name, role, section, score in ts.utterance_scores:
        utt_data.append({
            "Speaker": f"{name} ({role})",
            "Section": section,
            "Compound": score.compound,
            "Label": score.label,
            "Positive": score.positive,
            "Negative": score.negative,
        })
    df_utt = pd.DataFrame(utt_data)
    if not df_utt.empty:
        st.dataframe(df_utt, use_container_width=True)

    st.markdown("---")

    # Uncertainty
    st.subheader("Uncertainty / Hedge Language")
    scorer = get_scorer()
    unc_mgmt  = scorer.compute_uncertainty(t.management_text)
    unc_anlst = scorer.compute_uncertainty(t.analyst_text)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Management**")
        st.metric("Uncertainty Score", f"{unc_mgmt['uncertainty_score']:.3f}")
        st.metric("Hedge Word Density", f"{unc_mgmt['hedge_density']:.3f}")
    with col2:
        st.markdown("**Analysts**")
        st.metric("Uncertainty Score", f"{unc_anlst['uncertainty_score']:.3f}")
        st.metric("Hedge Word Density", f"{unc_anlst['hedge_density']:.3f}")


# ---------------------------------------------------------------------------
# Page 3 — Topic Modeling
# ---------------------------------------------------------------------------

elif page == "📚 Topic Modeling":
    st.title(f"📚 Topic Modeling — {ticker}")

    st.subheader(f"LDA Topics (n={topic_result.n_topics})")
    for topic in topic_result.topics:
        with st.expander(f"Topic {topic.topic_id}: {topic.label}  ({topic.doc_count} docs)"):
            kw_df = pd.DataFrame(topic.keywords, columns=["Keyword", "Weight"])
            st.dataframe(kw_df, use_container_width=True)
            st.caption(f"Representative: {topic.representative_text}")

    st.markdown("---")
    st.subheader("Top Keywords by Section")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Prepared Remarks**")
        prep_df = pd.DataFrame(
            section_kws.get("prepared_remarks", []),
            columns=["Keyword", "Count"]
        )
        st.dataframe(prep_df, use_container_width=True)

    with col2:
        st.markdown("**Q&A Section**")
        qa_df = pd.DataFrame(
            section_kws.get("qa", []),
            columns=["Keyword", "Count"]
        )
        st.dataframe(qa_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4 — Forward Guidance
# ---------------------------------------------------------------------------

elif page == "🔮 Forward Guidance":
    st.title(f"🔮 Forward Guidance — {ticker}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Statements", len(report.statements))
    with col2:
        st.metric("Quantitative", report.n_quantitative)
    with col3:
        color = "positive" if report.overall_sentiment > 0.1 else "negative" if report.overall_sentiment < -0.1 else "neutral"
        st.metric("Overall Sentiment", f"{report.overall_sentiment:+.3f}")

    st.markdown("---")

    # Surprise score
    st.subheader("Guidance Surprise Score")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Surprise Score", f"{surprise:+.3f}",
                  help="Positive = mgmt more bullish than analyst questions implied")
    with col2:
        signal = "🟢 LONG" if surprise > 0.1 else "🔴 SHORT" if surprise < -0.1 else "⚪ FLAT"
        st.metric("Signal", signal)

    st.markdown("---")

    # Guidance statements table
    st.subheader("Guidance Statements")
    if report.statements:
        stmt_df = pd.DataFrame([{
            "Speaker": s.speaker,
            "Category": s.category,
            "Sentiment": s.sentiment,
            "Score": f"{s.sentiment_score:+.3f}",
            "Quantitative": "✓" if s.is_quantitative else "",
            "Text": s.text[:100] + "..." if len(s.text) > 100 else s.text,
        } for s in report.statements])
        st.dataframe(stmt_df, use_container_width=True)

    # Category breakdown
    if report.sentiment_by_category:
        st.markdown("---")
        st.subheader("Sentiment by Category")
        cat_df = pd.DataFrame(
            list(report.sentiment_by_category.items()),
            columns=["Category", "Sentiment Score"]
        ).sort_values("Sentiment Score", ascending=False)
        st.bar_chart(cat_df.set_index("Category"))


# ---------------------------------------------------------------------------
# Page 5 — Signal & Backtest
# ---------------------------------------------------------------------------

elif page == "📈 Signal & Backtest":
    st.title("📈 Signal & Backtest Results")

    with st.spinner("Running backtest..."):
        bt_result = run_backtest()

    m = bt_result.metrics

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Sharpe Ratio", f"{m.get('sharpe', 0):.2f}")
    with col2:
        st.metric("Sortino Ratio", f"{m.get('sortino', 0):.2f}")
    with col3:
        st.metric("Max Drawdown", f"{m.get('max_drawdown', 0):.2f}%")
    with col4:
        st.metric("Win Rate", f"{m.get('win_rate', 0):.1f}%")
    with col5:
        st.metric("Total Trades", len(bt_result.trades))

    st.markdown("---")

    # Equity curve
    st.subheader("Equity Curve vs Benchmark")
    eq_df = pd.DataFrame({
        "Strategy":  bt_result.equity_curve.values,
        "Benchmark": bt_result.benchmark_curve.values,
    }, index=bt_result.equity_curve.index)
    st.line_chart(eq_df)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Sector performance
    with col1:
        st.subheader("Sector Performance")
        if not bt_result.sector_performance.empty:
            st.dataframe(bt_result.sector_performance, use_container_width=True)

    # Monthly returns
    with col2:
        st.subheader("Monthly Returns")
        if not bt_result.monthly_returns.empty:
            mt = (bt_result.monthly_returns * 100).round(2)
            st.dataframe(mt.style.background_gradient(cmap="RdYlGn", axis=None),
                         use_container_width=True)

    st.markdown("---")

    # Trade log
    st.subheader("Trade Log")
    if bt_result.trades:
        trade_df = pd.DataFrame([{
            "Ticker":    t.ticker,
            "Sector":    t.sector,
            "Direction": t.direction,
            "Sentiment": f"{t.sentiment_score:+.3f}",
            "PnL %":     f"{t.pnl_pct:+.3f}%",
            "Entry":     t.entry_date,
            "Exit":      t.exit_date,
        } for t in bt_result.trades])
        st.dataframe(trade_df, use_container_width=True)
