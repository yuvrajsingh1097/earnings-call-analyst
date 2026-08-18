"""
Unit Tests — Backtest Engine
==============================
Run with: python -m pytest tests/test_backtest_engine.py -v
"""

import pytest
import numpy as np
import pandas as pd
from backtest.engine import (
    compute_metrics, rolling_sharpe, monthly_returns_table,
    synthetic_universe, synthetic_signals, BacktestEngine,
    BacktestResult, Trade, SECTORS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tickers():
    return ["AAPL", "MSFT", "JPM", "JNJ", "XOM"]


@pytest.fixture(scope="module")
def universe(tickers):
    return synthetic_universe(tickers, n_days=504, seed=42)


@pytest.fixture(scope="module")
def signals(tickers, universe):
    return synthetic_signals(tickers, universe, n_events_per_ticker=4)


@pytest.fixture(scope="module")
def result(signals, universe):
    engine = BacktestEngine(threshold=0.15, hold_days=5)
    return engine.run(signals, universe)


@pytest.fixture(scope="module")
def sample_returns():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=252, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, 252), index=idx)


# ---------------------------------------------------------------------------
# 1. Risk metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_returns_dict(self, sample_returns):
        m = compute_metrics(sample_returns)
        assert isinstance(m, dict)

    def test_required_keys(self, sample_returns):
        m = compute_metrics(sample_returns)
        for k in ["sharpe", "sortino", "max_drawdown", "win_rate", "ann_return"]:
            assert k in m

    def test_max_drawdown_nonpositive(self, sample_returns):
        m = compute_metrics(sample_returns)
        assert m["max_drawdown"] <= 0

    def test_win_rate_in_range(self, sample_returns):
        m = compute_metrics(sample_returns)
        assert 0 <= m["win_rate"] <= 100

    def test_empty_returns(self):
        m = compute_metrics(pd.Series(dtype=float))
        assert m == {}

    def test_var_less_than_cvar(self, sample_returns):
        m = compute_metrics(sample_returns)
        assert m["var_95"] >= m["cvar_95"]

    def test_ann_vol_positive(self, sample_returns):
        m = compute_metrics(sample_returns)
        assert m["ann_vol"] > 0


# ---------------------------------------------------------------------------
# 2. Rolling Sharpe
# ---------------------------------------------------------------------------

class TestRollingSharpe:

    def test_returns_series(self, sample_returns):
        rs = rolling_sharpe(sample_returns, window=21)
        assert isinstance(rs, pd.Series)

    def test_same_length(self, sample_returns):
        rs = rolling_sharpe(sample_returns, window=21)
        assert len(rs) == len(sample_returns)

    def test_first_values_nan(self, sample_returns):
        rs = rolling_sharpe(sample_returns, window=21)
        assert rs.iloc[:20].isna().all()


# ---------------------------------------------------------------------------
# 3. Monthly returns table
# ---------------------------------------------------------------------------

class TestMonthlyReturns:

    def test_returns_dataframe(self, sample_returns):
        mt = monthly_returns_table(sample_returns)
        assert isinstance(mt, pd.DataFrame)

    def test_columns_are_months(self, sample_returns):
        mt = monthly_returns_table(sample_returns)
        valid = {"Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"}
        assert set(mt.columns).issubset(valid)

    def test_index_is_years(self, sample_returns):
        mt = monthly_returns_table(sample_returns)
        assert all(isinstance(y, (int, np.integer)) for y in mt.index)


# ---------------------------------------------------------------------------
# 4. Synthetic universe
# ---------------------------------------------------------------------------

class TestSyntheticUniverse:

    def test_returns_dict(self, universe):
        assert isinstance(universe, dict)

    def test_all_tickers_present(self, universe, tickers):
        for t in tickers:
            assert t in universe

    def test_correct_length(self, universe):
        for df in universe.values():
            assert len(df) == 504

    def test_positive_prices(self, universe):
        for df in universe.values():
            assert (df["Close"] > 0).all()

    def test_columns(self, universe):
        for df in universe.values():
            assert "Close" in df.columns
            assert "Return" in df.columns


# ---------------------------------------------------------------------------
# 5. Synthetic signals
# ---------------------------------------------------------------------------

class TestSyntheticSignals:

    def test_returns_list(self, signals):
        assert isinstance(signals, list)

    def test_signal_structure(self, signals):
        ticker, idx, sent, surp = signals[0]
        assert isinstance(ticker, str)
        assert isinstance(idx, (int, np.integer))
        assert -1 <= sent <= 1
        assert -1 <= surp <= 1

    def test_expected_count(self, signals, tickers):
        assert len(signals) == len(tickers) * 4


# ---------------------------------------------------------------------------
# 6. Backtest engine
# ---------------------------------------------------------------------------

class TestBacktestEngine:

    def test_returns_result(self, result):
        assert isinstance(result, BacktestResult)

    def test_has_trades(self, result):
        assert len(result.trades) > 0

    def test_trade_structure(self, result):
        t = result.trades[0]
        assert isinstance(t, Trade)
        assert t.direction in {"long", "short"}

    def test_equity_curve_series(self, result):
        assert isinstance(result.equity_curve, pd.Series)

    def test_equity_starts_at_capital(self, result):
        assert abs(result.equity_curve.iloc[0] - 100_000) < 1000

    def test_benchmark_series(self, result):
        assert isinstance(result.benchmark_curve, pd.Series)

    def test_metrics_dict(self, result):
        assert isinstance(result.metrics, dict)

    def test_sector_performance_df(self, result):
        assert isinstance(result.sector_performance, pd.DataFrame)

    def test_monthly_returns_df(self, result):
        assert isinstance(result.monthly_returns, pd.DataFrame)

    def test_rolling_sharpe_series(self, result):
        assert isinstance(result.rolling_sharpe, pd.Series)

    def test_pnl_reflects_costs(self, result):
        # All trades should have transaction cost applied
        for t in result.trades:
            assert t.transaction_cost > 0

    def test_high_threshold_fewer_trades(self, signals, universe):
        e1 = BacktestEngine(threshold=0.10)
        e2 = BacktestEngine(threshold=0.70)
        r1 = e1.run(signals, universe)
        r2 = e2.run(signals, universe)
        assert len(r1.trades) >= len(r2.trades)

    def test_sector_performance_columns(self, result):
        if not result.sector_performance.empty:
            for col in ["n_trades", "avg_pnl", "hit_rate"]:
                assert col in result.sector_performance.columns
