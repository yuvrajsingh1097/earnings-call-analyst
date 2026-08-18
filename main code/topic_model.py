"""
Topic Modeling — BERTopic + LDA Baseline
==========================================
Discovers key themes across earnings call transcripts using:
    1. BERTopic  — transformer-based topic modeling (primary)
    2. LDA       — Latent Dirichlet Allocation (baseline comparison)

Topics extracted per:
    - Full corpus (all transcripts)
    - Per section (prepared remarks vs Q&A)
    - Per speaker role (management vs analyst)

Outputs:
    - Topic labels + top keywords
    - Topic distribution per transcript
    - Topic evolution across quarters
    - BERTopic vs LDA comparison
"""

import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Topic:
    topic_id: int
    label: str              # auto-generated human-readable label
    keywords: list          # top N keywords with scores [(word, score)]
    doc_count: int          # number of utterances assigned to this topic
    representative_text: str


@dataclass
class TopicModelResult:
    model_type: str         # 'bertopic' or 'lda'
    topics: list            # list of Topic objects
    doc_topics: list        # topic assignment per document
    coherence: Optional[float]
    n_docs: int
    n_topics: int


# ---------------------------------------------------------------------------
# Text preprocessor for topic modeling
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "we", "our", "us", "i", "you", "they", "it",
    "he", "she", "its", "their", "your", "my", "his", "her", "also", "very",
    "just", "so", "than", "then", "about", "up", "out", "if", "when",
    "what", "which", "who", "how", "all", "each", "some", "such", "into",
    "through", "during", "well", "going", "think", "said", "thank", "good",
    "know", "see", "now", "get", "got", "one", "two", "three", "year",
    "quarter", "next", "last", "first", "second", "third", "fourth",
}

FINANCIAL_STOPWORDS = {
    "million", "billion", "percent", "basis", "points", "versus", "prior",
    "compared", "approximately", "company", "business", "results", "report",
    "reported", "reporting", "quarter", "fiscal", "annual",
}


def preprocess_for_topics(text: str, min_word_len: int = 3) -> list:
    """
    Tokenise and clean text for topic modeling.
    Returns list of meaningful tokens.
    """
    import re
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    tokens = text.split()
    tokens = [
        t for t in tokens
        if len(t) >= min_word_len
        and t not in STOPWORDS
        and t not in FINANCIAL_STOPWORDS
    ]
    return tokens


def build_corpus(transcripts: list, section: str = "all") -> list:
    """
    Build a list of text documents from transcripts for topic modeling.

    Parameters
    ----------
    transcripts : list of Transcript objects
    section     : 'all' | 'prepared_remarks' | 'qa'

    Returns list of (ticker, quarter, speaker_role, text) tuples.
    """
    docs = []
    for t in transcripts:
        for u in t.utterances:
            if u.speaker.role == "Operator" or u.word_count < 10:
                continue
            if section != "all" and u.section != section:
                continue
            docs.append({
                "ticker":  t.ticker,
                "quarter": t.quarter,
                "role":    u.speaker.role,
                "section": u.section,
                "text":    u.text,
                "tokens":  preprocess_for_topics(u.text),
            })
    return docs


# ---------------------------------------------------------------------------
# LDA baseline
# ---------------------------------------------------------------------------

class LDATopicModel:
    """
    Latent Dirichlet Allocation topic model using sklearn.
    Used as a fast baseline comparison against BERTopic.
    """

    def __init__(self, n_topics: int = 5, max_features: int = 500, seed: int = 42):
        self.n_topics   = n_topics
        self.max_features = max_features
        self.seed       = seed
        self.model      = None
        self.vectorizer = None

    def fit(self, docs: list) -> TopicModelResult:
        """
        Fit LDA on a list of document dicts (output of build_corpus).

        Returns TopicModelResult.
        """
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        texts = [" ".join(d["tokens"]) for d in docs if d["tokens"]]
        if len(texts) < 2:
            return self._empty_result()

        n_topics = min(self.n_topics, len(texts))

        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=1,
            max_df=0.95,
        )
        dtm = self.vectorizer.fit_transform(texts)

        self.model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=self.seed,
            max_iter=20,
        )
        doc_topic_matrix = self.model.fit_transform(dtm)
        doc_topics = doc_topic_matrix.argmax(axis=1).tolist()

        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for i, comp in enumerate(self.model.components_):
            top_idx     = comp.argsort()[-10:][::-1]
            keywords    = [(feature_names[j], round(comp[j] / comp.sum(), 4)) for j in top_idx]
            label       = " / ".join([k for k, _ in keywords[:3]])
            count       = int((np.array(doc_topics) == i).sum())
            rep_idx     = [j for j, dt in enumerate(doc_topics) if dt == i]
            rep_text    = docs[rep_idx[0]]["text"][:120] + "..." if rep_idx else ""
            topics.append(Topic(i, label, keywords, count, rep_text))

        return TopicModelResult(
            model_type="lda",
            topics=topics,
            doc_topics=doc_topics,
            coherence=None,
            n_docs=len(texts),
            n_topics=n_topics,
        )

    def _empty_result(self) -> TopicModelResult:
        return TopicModelResult("lda", [], [], None, 0, 0)

    def get_topic_distribution(self, text: str) -> np.ndarray:
        """Get topic distribution for a new text."""
        if self.model is None or self.vectorizer is None:
            return np.array([])
        tokens = preprocess_for_topics(text)
        vec = self.vectorizer.transform([" ".join(tokens)])
        return self.model.transform(vec)[0]


# ---------------------------------------------------------------------------
# BERTopic wrapper
# ---------------------------------------------------------------------------

class BERTopicModel:
    """
    BERTopic topic model.
    Falls back to LDA if bertopic/sentence-transformers unavailable.
    """

    def __init__(self, n_topics: int = 5, seed: int = 42):
        self.n_topics = n_topics
        self.seed     = seed
        self.model    = None
        self._has_bertopic = self._check_bertopic()

    def _check_bertopic(self) -> bool:
        try:
            import bertopic  # noqa
            return True
        except ImportError:
            return False

    def fit(self, docs: list) -> TopicModelResult:
        """
        Fit BERTopic on corpus. Falls back to LDA if unavailable.
        """
        if not self._has_bertopic:
            print("  BERTopic not available — using LDA fallback.")
            lda = LDATopicModel(n_topics=self.n_topics, seed=self.seed)
            result = lda.fit(docs)
            result.model_type = "lda(bertopic_fallback)"
            return result

        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer

        texts = [d["text"] for d in docs if d["tokens"]]
        if len(texts) < 4:
            lda = LDATopicModel(n_topics=min(self.n_topics, len(texts)))
            return lda.fit(docs)

        vectorizer = CountVectorizer(stop_words="english", min_df=1)
        try:
            topic_model = BERTopic(
                vectorizer_model=vectorizer,
                nr_topics=self.n_topics,
                calculate_probabilities=True,
                verbose=False,
            )
            topic_ids, probs = topic_model.fit_transform(texts)
            self.model = topic_model

            topics = []
            for tid in topic_model.get_topics():
                if tid == -1:
                    continue
                kws   = topic_model.get_topic(tid)
                label = " / ".join([w for w, _ in kws[:3]])
                count = int(sum(1 for t in topic_ids if t == tid))
                rep   = next((texts[i] for i, t in enumerate(topic_ids) if t == tid), "")
                topics.append(Topic(tid, label, kws[:10], count, rep[:120] + "..."))

            return TopicModelResult(
                model_type="bertopic",
                topics=topics,
                doc_topics=topic_ids,
                coherence=None,
                n_docs=len(texts),
                n_topics=len(topics),
            )
        except Exception as e:
            print(f"  BERTopic failed ({e}), using LDA.")
            lda = LDATopicModel(n_topics=self.n_topics, seed=self.seed)
            return lda.fit(docs)


# ---------------------------------------------------------------------------
# Topic analysis utilities
# ---------------------------------------------------------------------------

def topic_distribution_per_ticker(
    docs: list,
    doc_topics: list,
    tickers: list,
) -> pd.DataFrame:
    """
    Build a DataFrame showing topic distribution per ticker.
    Rows = tickers, Columns = topic IDs.
    """
    rows = []
    for ticker in tickers:
        ticker_topics = [
            dt for d, dt in zip(docs, doc_topics)
            if d["ticker"] == ticker
        ]
        if not ticker_topics:
            continue
        counter = Counter(ticker_topics)
        total   = len(ticker_topics)
        rows.append({
            "ticker": ticker,
            **{f"topic_{k}": round(v / total, 3) for k, v in counter.items()}
        })
    return pd.DataFrame(rows).fillna(0)


def top_keywords_per_section(docs: list, top_n: int = 10) -> dict:
    """
    Extract top N keywords per section (prepared_remarks vs qa)
    using simple TF weighting.
    """
    section_tokens = {"prepared_remarks": [], "qa": []}
    for d in docs:
        section_tokens[d["section"]].extend(d["tokens"])

    result = {}
    for section, tokens in section_tokens.items():
        counter = Counter(tokens)
        result[section] = counter.most_common(top_n)
    return result


def keyword_frequency_tracker(
    transcripts: list,
    keywords: list,
) -> pd.DataFrame:
    """
    Track frequency of specific keywords across tickers and quarters.

    Returns DataFrame: rows = (ticker, quarter), columns = keywords.
    """
    rows = []
    for t in transcripts:
        text   = t.raw_text.lower()
        words  = text.split()
        n      = max(len(words), 1)
        row    = {"ticker": t.ticker, "quarter": t.quarter}
        for kw in keywords:
            row[kw] = text.count(kw.lower()) / n * 1000  # per 1000 words
        rows.append(row)
    return pd.DataFrame(rows)


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

    print("=" * 60)
    print("Topic Modeling Demo")
    print("=" * 60)

    docs = build_corpus(transcripts, section="all")
    print(f"\n  Documents built : {len(docs)}")

    # LDA
    print("\n  Fitting LDA...")
    lda = LDATopicModel(n_topics=3)
    lda_result = lda.fit(docs)
    print(f"  Topics found    : {lda_result.n_topics}")
    for t in lda_result.topics:
        kws = ", ".join([k for k, _ in t.keywords[:5]])
        print(f"    Topic {t.topic_id}: {kws}  ({t.doc_count} docs)")

    # BERTopic (fallback to LDA)
    print("\n  Fitting BERTopic (with fallback)...")
    bert = BERTopicModel(n_topics=3)
    bert_result = bert.fit(docs)
    print(f"  Model used      : {bert_result.model_type}")
    print(f"  Topics found    : {bert_result.n_topics}")

    # Keyword tracker
    keywords = ["revenue", "growth", "margin", "cloud", "ai", "services"]
    kw_df = keyword_frequency_tracker(transcripts, keywords)
    print("\n  Keyword frequency (per 1000 words):")
    print(kw_df.to_string(index=False))

    # Section keywords
    section_kws = top_keywords_per_section(docs)
    print("\n  Top keywords — Prepared Remarks:")
    print("   ", [w for w, _ in section_kws["prepared_remarks"][:8]])
    print("  Top keywords — Q&A:")
    print("   ", [w for w, _ in section_kws["qa"][:8]])
