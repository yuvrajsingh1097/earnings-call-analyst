"""
Unit Tests — Price Data & Sentiment → Return Mapping
======================================================
Run with: python -m pytest tests/test_alpha_gen.py -v
"""

import pytest
import numpy as np
import pandas as pd
from signals.alpha_gen import (
    synthetic_prices, get_market_returns,
    EarningsEvent, EventStudy, EventStudyResult,
    sentiment_return_correlation, backtest_sentiment_signal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def price_data():
    return {
        "AAPL": synthetic_prices("AAPL", n_days=504, earnings_dates=[(120, 0.5)]),
        "MSFT": synthetic_prices("MSFT", n_days=504, earnings_dates=[(180, 0.4)]),
    }


@pytest.fixture(scope="module")
def market_rets():
    return get_market_returns(n_days=504)


@pytest.fixture(scope="module")
def events(price_data):
    aapl_date = str(price_data["AAPL"].index[120].date())
    msft_date = str(price_data["MSFT"].index[180].date())
    return [
        EarningsEvent("AAPL", "Q3 2024", aapl_date,  0.50, -0.17),
        EarningsEvent("MSFT", "Q4 2024", msft_date,  0.45, -0.33),
        EarningsEvent("AAPL", "Q2 2024", aapl_date,  -0.30, -0.20),
    ]


@pytest.fixture(scope="module")
def study_result(events, price_data, market_rets):
    study = EventStudy(window=5)
    return study, study.run(events, price_data, market_rets)


# ---------------------------------------------------------------------------
# 1. Synthetic price data
# ---------------------------------------------------------------------------

class TestSyntheticPrices:

    def test_returns_dataframe(self):
        df = synthetic_prices("TEST", n_days=100)
        assert isinstance(df, pd.DataFrame)

    def test_correct_length(self):
        df = synthetic_prices("TEST", n_days=100)
        assert len(df) == 100

    def test_positive_prices(self):
        df = synthetic_prices("TEST", n_days=200)
        assert (df["Close"] > 0).all()

    def test_columns_present(self):
        df = synthetic_prices("TEST", n_days=100)
        for col in ["Close", "Return", "Volume"]:
            assert col in df.columns

    def test_earnings_jump_applied(self):
        df_flat = synthetic_prices("TEST", n_days=200, seed=0)
        df_jump = synthetic_prices("TEST", n_days=200,
                                   earnings_dates=[(100, 1.0)], seed=0)
        assert df_flat["Return"].iloc[100] != df_jump["Return"].iloc[100]

    def test_market_returns_series(self):
        mkt = get_market_returns(n_days=200)
        assert isinstance(mkt, pd.Series)
        assert len(mkt) == 200


# ---------------------------------------------------------------------------
# 2. EarningsEvent
# ---------------------------------------------------------------------------

class TestEarningsEvent:

    def test_creation(self):
        e = EarningsEvent("AAPL", "Q3 2024", "2024-08-01", 0.5, -0.2)
        assert e.ticker == "AAPL"
        assert e.sentiment_score == 0.5

    def test_defaults_none(self):
        e = EarningsEvent("AAPL", "Q3 2024", "2024-08-01", 0.5, -0.2)
        assert e.next_day_return is None
        assert e.car_minus5_plus5 is None


# ---------------------------------------------------------------------------
# 3. EventStudy
# ---------------------------------------------------------------------------

class TestEventStudy:

    def test_returns_result(self, study_result):
        _, result = study_result
        assert isinstance(result, EventStudyResult)

    def test_n_events_positive(self, study_result):
        _, result = study_result
        assert result.n_events > 0

    def test_event_window_length(self, study_result):
        study, result = study_result
        assert len(result.event_window) == 2 * study.window + 1

    def test_avg_car_length(self, study_result):
        study, result = study_result
        assert len(result.avg_car) == 2 * study.window + 1

    def test_hit_rate_in_range(self, study_result):
        _, result = study_result
        assert 0.0 <= result.hit_rate <= 1.0

    def test_events_populate_returns(self, events, price_data, market_rets):
        study = EventStudy(window=5)
        study.run(events, price_data, market_rets)
        filled = [e for e in events if e.next_day_return is not None]
        assert len(filled) > 0

    def test_car_values_finite(self, study_result):
        _, result = study_result
        assert all(np.isfinite(v) for v in result.avg_car)

    def test_unknown_ticker_skipped(self, market_rets):
        bad_events = [EarningsEvent("ZZZZ", "Q1 2024", "2024-01-01", 0.5, 0.1)]
        study = EventStudy(window=5)
        result = study.run(bad_events, {}, market_rets)
        assert result.n_events == 0

    def test_compute_car_returns_array(self, price_data, market_rets):
        study = EventStudy(window=5)
        rets  = price_data["AAPL"]["Return"]
        date  = price_data["AAPL"].index[120]
        car   = study.compute_car(rets, market_rets, date)
        assert car is not None
        assert len(car) == 2 * study.window + 1


# ---------------------------------------------------------------------------
# 4. Sentiment → return correlation
# ---------------------------------------------------------------------------

class TestSentimentCorrelation:

    def test_returns_dataframe(self, events, price_data, market_rets):
        study = EventStudy(window=5)
        study.run(events, price_data, market_rets)
        df = sentiment_return_correlation(events)
        assert isinstance(df, pd.DataFrame)

    def test_correlation_in_range(self, events, price_data, market_rets):
        study = EventStudy(window=5)
        study.run(events, price_data, market_rets)
        df = sentiment_return_correlation(events)
        if not df.empty:
            for val in df.iloc[0].values:
                assert -1.0 <= val <= 1.0

    def test_empty_events_returns_empty(self):
        df = sentiment_return_correlation([])
        assert df.empty


# ---------------------------------------------------------------------------
# 5. Signal backtest
# ---------------------------------------------------------------------------

class TestSignalBacktest:

    def test_returns_dataframe(self, events, price_data, market_rets):
        study = EventStudy(window=5)
        study.run(events, price_data, market_rets)
        df = backtest_sentiment_signal(events)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self, events, price_data, market_rets):
        study = EventStudy(window=5)
        study.run(events, price_data, market_rets)
        df = backtest_sentiment_signal(events)
        if not df.empty:
            for col in ["ticker", "direction", "pnl"]:
                assert col in df.columns

    def test_direction_values(self, events, price_data, market_rets):
        study = EventStudy(window=5)
        study.run(events, price_data, market_rets)
        df = backtest_sentiment_signal(events)
        if not df.empty:
            assert df["direction"].isin(["long", "short", "flat"]).all()

    def test_high_threshold_all_flat(self, events, price_data, market_rets):
        study = EventStudy(window=5)
        study.run(events, price_data, market_rets)
        df = backtest_sentiment_signal(events, threshold=999)
        if not df.empty:
            assert (df["direction"] == "flat").all()
