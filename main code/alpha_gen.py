"""
Price Data Loader & Sentiment → Return Mapping
================================================
Loads price data aligned to earnings dates and computes:
    1. Abnormal returns around earnings (event study)
    2. Sentiment score → next-day / next-week return correlation
    3. Cumulative Abnormal Returns (CAR) for t-5 to t+5 window
    4. Signal backtest: long/short based on sentiment surprise

Uses synthetic price data when yfinance is unavailable.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EarningsEvent:
    ticker: str
    quarter: str
    date: str                    # earnings date (YYYY-MM-DD)
    sentiment_score: float       # overall guidance sentiment
    surprise_score: float        # guidance surprise vs analyst
    next_day_return: Optional[float] = None
    next_week_return: Optional[float] = None
    car_minus5_plus5: Optional[float] = None   # cumulative abnormal return


@dataclass
class EventStudyResult:
    ticker: str
    event_window: list           # [-5, -4, ..., 0, ..., +5]
    avg_car: list                # average CAR across all events
    n_events: int
    hit_rate: float              # % of correct signal direction
    avg_next_day_return: float
    sharpe: float


# ---------------------------------------------------------------------------
# Synthetic price generator
# ---------------------------------------------------------------------------

def synthetic_prices(
    ticker: str,
    n_days: int = 504,
    earnings_dates: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily OHLCV with earnings-day jumps.

    Parameters
    ----------
    ticker         : ticker symbol
    n_days         : number of trading days
    earnings_dates : list of (date_index, sentiment) tuples for earnings jumps
    seed           : random seed
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")

    mu    = 0.08 / 252
    sigma = 0.20 / np.sqrt(252)

    log_rets = mu + sigma * rng.standard_normal(n_days)

    # Add earnings-day jumps based on sentiment
    if earnings_dates:
        for idx, sentiment in earnings_dates:
            if 0 <= idx < n_days:
                jump = sentiment * 0.03 + rng.standard_normal() * 0.01
                log_rets[idx] += jump

    close = 100.0 * np.exp(np.cumsum(log_rets))
    volume = rng.integers(1_000_000, 5_000_000, n_days).astype(float)

    return pd.DataFrame({
        "Date":   dates,
        "Close":  close,
        "Return": log_rets,
        "Volume": volume,
    }).set_index("Date")


def get_market_returns(n_days: int = 504, seed: int = 99) -> pd.Series:
    """Generate synthetic market (SPY) returns for abnormal return computation."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    mu    = 0.07 / 252
    sigma = 0.15 / np.sqrt(252)
    rets  = mu + sigma * rng.standard_normal(n_days)
    return pd.Series(rets, index=dates, name="market_return")


# ---------------------------------------------------------------------------
# Event study engine
# ---------------------------------------------------------------------------

class EventStudy:
    """
    Computes Cumulative Abnormal Returns (CAR) around earnings events.

    CAR_t = Σ (R_it - R_mt)  for t in event window

    Where:
        R_it = stock return on day t
        R_mt = market return on day t (benchmark)
    """

    def __init__(self, window: int = 5):
        self.window = window   # days before and after event

    def compute_car(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        event_date,
    ) -> Optional[np.ndarray]:
        """
        Compute CAR for a single event.

        Returns array of length (2*window + 1) or None if data insufficient.
        """
        abnormal = stock_returns - market_returns.reindex(stock_returns.index).fillna(0)

        try:
            event_loc = abnormal.index.get_loc(event_date)
        except KeyError:
            # Find nearest date
            nearest = abnormal.index[np.argmin(np.abs(abnormal.index - pd.Timestamp(event_date)))]
            event_loc = abnormal.index.get_loc(nearest)

        start = event_loc - self.window
        end   = event_loc + self.window + 1

        if start < 0 or end > len(abnormal):
            return None

        window_rets = abnormal.iloc[start:end].values
        car = np.cumsum(window_rets)
        return car

    def run(
        self,
        events: list,
        price_data: dict,
        market_returns: pd.Series,
    ) -> EventStudyResult:
        """
        Run event study across all earnings events.

        Parameters
        ----------
        events        : list of EarningsEvent objects
        price_data    : dict of {ticker: price DataFrame}
        market_returns: market return Series

        Returns EventStudyResult with average CAR and signal metrics.
        """
        all_cars  = []
        directions_correct = []
        next_day_returns   = []

        for event in events:
            if event.ticker not in price_data:
                continue

            prices = price_data[event.ticker]
            rets   = prices["Return"]

            car = self.compute_car(rets, market_returns, event.date)
            if car is None:
                continue

            all_cars.append(car)

            # Next-day return (day +1)
            next_day = float(car[self.window + 1] - car[self.window]) if len(car) > self.window + 1 else 0
            event.next_day_return  = round(next_day, 6)
            event.car_minus5_plus5 = round(float(car[-1]), 6)
            next_day_returns.append(next_day)

            # Signal direction: positive sentiment → expect positive return
            if event.sentiment_score != 0:
                correct = (event.sentiment_score > 0) == (next_day > 0)
                directions_correct.append(correct)

        if not all_cars:
            return EventStudyResult(
                ticker="ALL", event_window=list(range(-self.window, self.window+1)),
                avg_car=[0.0]*(2*self.window+1), n_events=0,
                hit_rate=0.0, avg_next_day_return=0.0, sharpe=0.0,
            )

        avg_car  = np.mean(all_cars, axis=0).tolist()
        hit_rate = float(np.mean(directions_correct)) if directions_correct else 0.0
        avg_ret  = float(np.mean(next_day_returns)) if next_day_returns else 0.0
        sharpe   = float(avg_ret / (np.std(next_day_returns) + 1e-8)) * np.sqrt(252) \
                   if len(next_day_returns) > 1 else 0.0

        return EventStudyResult(
            ticker="ALL",
            event_window=list(range(-self.window, self.window+1)),
            avg_car=avg_car,
            n_events=len(all_cars),
            hit_rate=round(hit_rate, 4),
            avg_next_day_return=round(avg_ret, 6),
            sharpe=round(sharpe, 4),
        )


# ---------------------------------------------------------------------------
# Sentiment → return correlation
# ---------------------------------------------------------------------------

def sentiment_return_correlation(events: list) -> pd.DataFrame:
    """
    Compute correlation between sentiment scores and forward returns.

    Returns DataFrame with correlation stats.
    """
    rows = [
        {
            "ticker":           e.ticker,
            "quarter":          e.quarter,
            "sentiment_score":  e.sentiment_score,
            "surprise_score":   e.surprise_score,
            "next_day_return":  e.next_day_return,
            "car_t5":           e.car_minus5_plus5,
        }
        for e in events
        if e.next_day_return is not None
    ]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    correlations = {}
    for signal in ["sentiment_score", "surprise_score"]:
        for ret in ["next_day_return", "car_t5"]:
            if df[signal].std() > 0 and df[ret].std() > 0:
                corr = df[signal].corr(df[ret])
                correlations[f"{signal}_vs_{ret}"] = round(float(corr), 4)

    return pd.DataFrame([correlations])


# ---------------------------------------------------------------------------
# Signal backtest
# ---------------------------------------------------------------------------

def backtest_sentiment_signal(
    events: list,
    threshold: float = 0.1,
    hold_days: int = 1,
) -> pd.DataFrame:
    """
    Simple signal backtest:
        signal > threshold  → long  (buy next open, sell after hold_days)
        signal < -threshold → short (sell short, cover after hold_days)
        otherwise           → flat

    Returns trade log DataFrame.
    """
    trades = []
    for e in events:
        if e.next_day_return is None:
            continue

        if e.sentiment_score > threshold:
            direction = "long"
            pnl = e.next_day_return
        elif e.sentiment_score < -threshold:
            direction = "short"
            pnl = -e.next_day_return
        else:
            direction = "flat"
            pnl = 0.0

        trades.append({
            "ticker":    e.ticker,
            "quarter":   e.quarter,
            "signal":    round(e.sentiment_score, 4),
            "direction": direction,
            "pnl":       round(float(pnl), 6),
        })

    df = pd.DataFrame(trades)
    if df.empty:
        return df

    active = df[df["direction"] != "flat"].copy()
    if active.empty:
        return df

    active["cumulative_pnl"] = active["pnl"].cumsum()
    total_ret = float(active["pnl"].sum())
    hit_rate  = float((active["pnl"] > 0).mean())
    sharpe    = float(active["pnl"].mean() / (active["pnl"].std() + 1e-8)) * np.sqrt(252)

    print(f"  Backtest summary:")
    print(f"    Trades     : {len(active)}")
    print(f"    Total PnL  : {total_ret:+.4f}")
    print(f"    Hit rate   : {hit_rate:.1%}")
    print(f"    Sharpe     : {sharpe:.2f}")

    return df


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from scraper.transcript_loader import load_synthetic_transcript
    from nlp.finbert_scorer import FinBERTScorer
    from nlp.guidance_extractor import GuidanceExtractor, guidance_surprise_score

    print("=" * 60)
    print("Price Data & Sentiment → Return Mapping Demo")
    print("=" * 60)

    transcripts = [
        load_synthetic_transcript("AAPL"),
        load_synthetic_transcript("MSFT"),
    ]

    scorer    = FinBERTScorer(use_transformers=False)
    extractor = GuidanceExtractor()

    # Build events list
    events = []
    for t in transcripts:
        ts      = scorer.score_transcript(t)
        report  = extractor.extract(t)
        surprise = guidance_surprise_score(
            ts.prepared_sentiment.compound,
            ts.analyst_sentiment.compound if ts.analyst_sentiment else 0,
        )
        events.append(EarningsEvent(
            ticker=t.ticker, quarter=t.quarter,
            date=t.date,
            sentiment_score=ts.overall.compound,
            surprise_score=surprise,
        ))

    # Synthetic price data
    earnings_map = {
        "AAPL": [(120, events[0].sentiment_score)],
        "MSFT": [(180, events[1].sentiment_score)],
    }
    price_data = {
        t: synthetic_prices(t, n_days=504, earnings_dates=earnings_map[t])
        for t in ["AAPL", "MSFT"]
    }
    market_rets = get_market_returns(n_days=504)

    # Event study
    study = EventStudy(window=5)
    result = study.run(events, price_data, market_rets)

    print(f"\nEvent Study:")
    print(f"  Events analysed : {result.n_events}")
    print(f"  Hit rate        : {result.hit_rate:.1%}")
    print(f"  Avg next-day ret: {result.avg_next_day_return:+.4f}")
    print(f"  Sharpe (ann.)   : {result.sharpe:.2f}")

    # Correlations
    corr_df = sentiment_return_correlation(events)
    if not corr_df.empty:
        print(f"\nSentiment → Return Correlations:")
        for col, val in corr_df.iloc[0].items():
            print(f"  {col:<35}: {val:+.4f}")

    # Backtest
    print(f"\nSignal Backtest (threshold=0.1):")
    trade_df = backtest_sentiment_signal(events)
