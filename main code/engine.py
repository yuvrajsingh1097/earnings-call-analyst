"""
Backtest Engine
================
Full backtest of the earnings sentiment signal across a stock universe.

Features:
    1. Multi-stock signal backtest with transaction costs
    2. Risk metrics: Sharpe, Sortino, Calmar, Max Drawdown, VaR, CVaR
    3. Sector-wise performance breakdown
    4. Monthly returns heatmap data
    5. Benchmark comparison (buy-and-hold)
    6. Rolling Sharpe ratio
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
class Trade:
    ticker: str
    sector: str
    entry_date: str
    exit_date: str
    direction: str          # long | short
    sentiment_score: float
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_dollar: float
    transaction_cost: float


@dataclass
class BacktestResult:
    trades: list
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    metrics: dict
    monthly_returns: pd.DataFrame
    sector_performance: pd.DataFrame
    rolling_sharpe: pd.Series


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def compute_metrics(returns: pd.Series, risk_free: float = 0.05) -> dict:
    """
    Compute full suite of risk-adjusted performance metrics.

    Parameters
    ----------
    returns    : daily return series
    risk_free  : annual risk-free rate

    Returns dict with all metrics.
    """
    if returns.empty or returns.std() == 0:
        return {}

    rf_daily = risk_free / 252
    excess   = returns - rf_daily

    # Return metrics
    total_return  = float((1 + returns).prod() - 1)
    ann_return    = float((1 + total_return) ** (252 / max(len(returns), 1)) - 1)
    ann_vol       = float(returns.std() * np.sqrt(252))

    # Sharpe
    sharpe = float(excess.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Sortino (downside deviation only)
    downside = returns[returns < rf_daily]
    sortino  = float(excess.mean() / downside.std() * np.sqrt(252)) \
               if len(downside) > 1 and downside.std() > 0 else 0

    # Max Drawdown
    cum_ret  = (1 + returns).cumprod()
    roll_max = cum_ret.cummax()
    drawdown = (cum_ret - roll_max) / roll_max
    max_dd   = float(drawdown.min())

    # Calmar
    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0

    # VaR and CVaR (95%)
    var_95  = float(np.percentile(returns, 5))
    cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95

    # Win rate
    win_rate = float((returns > 0).mean())

    # Best / worst day
    best_day  = float(returns.max())
    worst_day = float(returns.min())

    return {
        "total_return":  round(total_return * 100, 2),
        "ann_return":    round(ann_return * 100, 2),
        "ann_vol":       round(ann_vol * 100, 2),
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "calmar":        round(calmar, 3),
        "max_drawdown":  round(max_dd * 100, 2),
        "var_95":        round(var_95 * 100, 4),
        "cvar_95":       round(cvar_95 * 100, 4),
        "win_rate":      round(win_rate * 100, 2),
        "best_day":      round(best_day * 100, 4),
        "worst_day":     round(worst_day * 100, 4),
        "n_days":        len(returns),
    }


def rolling_sharpe(returns: pd.Series, window: int = 63, rf: float = 0.05) -> pd.Series:
    """Compute rolling Sharpe ratio."""
    rf_daily = rf / 252
    excess   = returns - rf_daily
    roll_mean = excess.rolling(window).mean()
    roll_std  = returns.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(252)).rename("rolling_sharpe")


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Build a Year × Month table of returns (for heatmap).

    Returns DataFrame with years as index, months (1–12) as columns.
    """
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df = monthly.to_frame("return")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="return")
    pivot.columns = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ][:len(pivot.columns)]
    return pivot.round(4)


# ---------------------------------------------------------------------------
# Synthetic universe generator
# ---------------------------------------------------------------------------

SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "JPM":  "Financials",  "BAC":  "Financials",
    "JNJ":  "Healthcare",  "PFE":  "Healthcare",
    "XOM":  "Energy",      "CVX":  "Energy",
    "WMT":  "Consumer",    "AMZN": "Consumer",
}


def synthetic_universe(
    tickers: list,
    n_days: int = 504,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic price data for a universe of stocks.
    Each stock has sector-correlated returns + idiosyncratic noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")

    # Sector return factors
    sector_rets = {
        "Technology": rng.standard_normal(n_days) * 0.012,
        "Financials":  rng.standard_normal(n_days) * 0.009,
        "Healthcare":  rng.standard_normal(n_days) * 0.007,
        "Energy":      rng.standard_normal(n_days) * 0.011,
        "Consumer":    rng.standard_normal(n_days) * 0.008,
    }

    universe = {}
    for ticker in tickers:
        sector = SECTORS.get(ticker, "Technology")
        factor = sector_rets.get(sector, sector_rets["Technology"])
        idio   = rng.standard_normal(n_days) * 0.008
        rets   = 0.0003 + 0.6 * factor + 0.4 * idio  # drift + sector + idio
        close  = 100 * np.exp(np.cumsum(rets))
        universe[ticker] = pd.DataFrame({
            "Close": close, "Return": rets
        }, index=dates)

    return universe


def synthetic_signals(
    tickers: list,
    price_data: dict,
    n_events_per_ticker: int = 4,
    seed: int = 7,
) -> list:
    """
    Generate synthetic earnings sentiment signals for each ticker.
    Returns list of (ticker, date_idx, sentiment_score, surprise_score) tuples.
    """
    rng = np.random.default_rng(seed)
    signals = []

    for ticker in tickers:
        prices = price_data[ticker]
        n      = len(prices)
        # Spread events evenly across the year
        spacing = n // (n_events_per_ticker + 1)
        for i in range(n_events_per_ticker):
            idx  = spacing * (i + 1) + rng.integers(-5, 5)
            idx  = max(10, min(n - 10, idx))
            sent = float(rng.uniform(-0.8, 0.8))
            surp = float(rng.uniform(-0.5, 0.5))
            signals.append((ticker, idx, sent, surp))

    return signals


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Runs the full earnings sentiment signal backtest.

    Strategy:
        On each earnings date, if sentiment > threshold → long
        If sentiment < -threshold → short
        Hold for `hold_days`, then exit.
        Apply transaction_cost to each trade.
    """

    def __init__(
        self,
        threshold: float = 0.15,
        hold_days: int = 5,
        transaction_cost: float = 0.001,
        initial_capital: float = 100_000,
        position_size: float = 0.10,   # 10% of capital per trade
    ):
        self.threshold        = threshold
        self.hold_days        = hold_days
        self.transaction_cost = transaction_cost
        self.initial_capital  = initial_capital
        self.position_size    = position_size

    def run(
        self,
        signals: list,
        price_data: dict,
        sectors: dict = None,
    ) -> BacktestResult:
        """
        Run backtest over all signals.

        Parameters
        ----------
        signals    : list of (ticker, date_idx, sentiment, surprise) tuples
        price_data : dict of {ticker: DataFrame with Close, Return}
        sectors    : dict of {ticker: sector_name}

        Returns BacktestResult.
        """
        if sectors is None:
            sectors = SECTORS

        trades = []
        all_dates = list(price_data.values())[0].index
        daily_pnl = pd.Series(0.0, index=all_dates)

        for ticker, date_idx, sentiment, surprise in signals:
            if ticker not in price_data:
                continue

            prices = price_data[ticker]
            n      = len(prices)

            # Skip if not enough data after event
            if date_idx + self.hold_days >= n:
                continue

            # Signal direction
            if sentiment > self.threshold:
                direction = "long"
            elif sentiment < -self.threshold:
                direction = "short"
            else:
                continue  # flat — no trade

            entry_price = float(prices["Close"].iloc[date_idx])
            exit_price  = float(prices["Close"].iloc[date_idx + self.hold_days])

            raw_ret = (exit_price - entry_price) / entry_price
            if direction == "short":
                raw_ret = -raw_ret

            cost    = self.transaction_cost * 2   # round-trip
            net_ret = raw_ret - cost
            capital = self.initial_capital * self.position_size
            pnl_dollar = capital * net_ret

            # Book PnL on exit day
            exit_date = prices.index[date_idx + self.hold_days]
            if exit_date in daily_pnl.index:
                daily_pnl[exit_date] += net_ret * self.position_size

            trades.append(Trade(
                ticker=ticker,
                sector=sectors.get(ticker, "Unknown"),
                entry_date=str(prices.index[date_idx].date()),
                exit_date=str(exit_date.date()),
                direction=direction,
                sentiment_score=round(sentiment, 4),
                entry_price=round(entry_price, 2),
                exit_price=round(exit_price, 2),
                pnl_pct=round(net_ret * 100, 4),
                pnl_dollar=round(pnl_dollar, 2),
                transaction_cost=round(cost * 100, 4),
            ))

        # Build equity curve
        equity = (1 + daily_pnl).cumprod() * self.initial_capital
        benchmark_rets = list(price_data.values())[0]["Return"]
        benchmark = (1 + benchmark_rets).cumprod() * self.initial_capital

        # Metrics
        strategy_rets = daily_pnl
        metrics = compute_metrics(strategy_rets[strategy_rets != 0])

        # Monthly returns
        monthly = monthly_returns_table(strategy_rets)

        # Sector breakdown
        trade_df = pd.DataFrame([{
            "sector": t.sector,
            "pnl_pct": t.pnl_pct,
            "direction": t.direction,
        } for t in trades])

        if not trade_df.empty:
            sector_perf = trade_df.groupby("sector").agg(
                n_trades   =("pnl_pct", "count"),
                avg_pnl    =("pnl_pct", "mean"),
                total_pnl  =("pnl_pct", "sum"),
                hit_rate   =("pnl_pct", lambda x: (x > 0).mean() * 100),
            ).round(3)
        else:
            sector_perf = pd.DataFrame()

        roll_sharpe = rolling_sharpe(strategy_rets)

        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            benchmark_curve=benchmark,
            metrics=metrics,
            monthly_returns=monthly,
            sector_performance=sector_perf,
            rolling_sharpe=roll_sharpe,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Backtest Engine Demo")
    print("=" * 60)

    tickers = list(SECTORS.keys())
    universe = synthetic_universe(tickers, n_days=504)
    signals  = synthetic_signals(tickers, universe, n_events_per_ticker=4)

    engine = BacktestEngine(threshold=0.15, hold_days=5, transaction_cost=0.001)
    result = engine.run(signals, universe)

    print(f"\n  Trades executed : {len(result.trades)}")
    print(f"\n  Performance Metrics:")
    for k, v in result.metrics.items():
        print(f"    {k:<20}: {v}")

    print(f"\n  Sector Performance:")
    print(result.sector_performance.to_string())

    print(f"\n  Monthly Returns (last 6 months):")
    print(result.monthly_returns.iloc[-1:].to_string())
