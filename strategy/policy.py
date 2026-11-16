"""Position sizing, rebalance rules, and performance math.

Trades are at most weekly, aimed at monthly, except a real-time cut when a
held index or the market breaks. The shock rule only reduces risk.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from strategy.universe import (
    CASH,
    MAX_WEIGHT,
    MIN_TURNOVER,
    SHOCK_FIVE_DAY,
    SHOCK_GROSS,
    SHOCK_ONE_DAY,
    TOP_HOLDINGS,
    TRADE_COST,
    WEEKLY_DRIFT,
)


def turnover(current: Mapping[str, float], target: Mapping[str, float]) -> float:
    keys = set(current) | set(target)
    return float(sum(abs(current.get(key, 0.0) - target.get(key, 0.0)) for key in keys))


def drift(weights: Mapping[str, float], returns: Mapping[str, float]) -> dict[str, float]:
    grown = {key: weight * (1.0 + float(returns.get(key, 0.0) or 0.0)) for key, weight in weights.items()}
    total = sum(grown.values())
    if total <= 0:
        return {key: (1.0 if key == CASH else 0.0) for key in weights}
    return {key: value / total for key, value in grown.items()}


def allocate(
    mu: Mapping[str, float],
    ann_vol: Mapping[str, float],
    risk_aversion: float,
    cap: float = MAX_WEIGHT,
    cash: str = CASH,
) -> dict[str, float]:
    """Long-only weights. Positive 21-day excess return earns risk budget; the rest is cash."""
    horizon = math.sqrt(21.0 / 252.0)
    scores: dict[str, float] = {}
    for ticker, raw in mu.items():
        if ticker == cash or raw is None or not math.isfinite(raw) or raw <= 0:
            continue
        vol = ann_vol.get(ticker, 0.2)
        if vol is None or not math.isfinite(vol):
            vol = 0.2
        vol_h = max(float(vol), 0.05) * horizon
        scores[ticker] = float(raw) / vol_h

    if len(scores) > TOP_HOLDINGS:
        keep = sorted(scores, key=scores.get, reverse=True)[:TOP_HOLDINGS]
        scores = {ticker: scores[ticker] for ticker in keep}

    names = set(mu) | set(scores) | {cash}
    weights = {ticker: 0.0 for ticker in names}
    if not scores or risk_aversion <= 0:
        weights[cash] = 1.0
        return weights

    mean_score = sum(scores.values()) / len(scores)
    gross = max(0.0, min(1.0, mean_score / risk_aversion))
    total = sum(scores.values())
    for ticker, score in scores.items():
        weights[ticker] = gross * score / total
    for ticker in list(weights):
        if ticker != cash and weights[ticker] > cap:
            weights[ticker] = cap
    risky = sum(value for ticker, value in weights.items() if ticker != cash)
    weights[cash] = max(0.0, 1.0 - risky)
    scale = sum(weights.values())
    return {ticker: value / scale for ticker, value in weights.items()}


def apply_shock(weights: Mapping[str, float], gross_cap: float = SHOCK_GROSS, cash: str = CASH) -> dict[str, float]:
    """Cut risky exposure to `gross_cap`. Never adds risk."""
    out = {ticker: float(weight) for ticker, weight in weights.items()}
    if cash not in out:
        out[cash] = 0.0
    risky = sum(value for ticker, value in out.items() if ticker != cash)
    if risky <= gross_cap + 1e-12:
        total = sum(out.values()) or 1.0
        return {ticker: value / total for ticker, value in out.items()}
    scale = gross_cap / risky
    for ticker in list(out):
        if ticker != cash:
            out[ticker] *= scale
    out[cash] = max(0.0, 1.0 - sum(value for ticker, value in out.items() if ticker != cash))
    total = sum(out.values())
    return {ticker: value / total for ticker, value in out.items()}


def decide(
    today: pd.Timestamp,
    last_trade: pd.Timestamp | None,
    last_reason: str | None,
    current: Mapping[str, float],
    desired: Mapping[str, float],
    shock: bool,
    shock_entry: bool = False,
) -> tuple[bool, str, dict[str, float]]:
    """Return whether to trade, why, and the weights to hold after the close.

    A shock trades once, on the day it starts. The book then stays put until
    the shock has cleared and the monthly or weekly rule allows a trade.
    """
    del last_reason  # kept so callers can pass the previous reason for the log
    desired = {key: float(value) for key, value in desired.items()}
    current = {key: float(value) for key, value in current.items()}
    if last_trade is None:
        return True, "initial", desired

    days = int((pd.Timestamp(today) - pd.Timestamp(last_trade)).days)
    moved = turnover(current, desired)
    if shock and not shock_entry:
        return False, "hold_shock", dict(current)
    if shock and shock_entry:
        if moved < MIN_TURNOVER or days < 1:
            return False, "hold_shock", dict(current)
        return True, "shock", desired
    if days < 7:
        return False, "hold_weekly_lock", dict(current)
    today = pd.Timestamp(today)
    last = pd.Timestamp(last_trade)
    new_month = (today.year, today.month) != (last.year, last.month)
    if new_month and moved >= MIN_TURNOVER:
        return True, "monthly", desired
    if moved >= WEEKLY_DRIFT:
        return True, "weekly", desired
    if new_month:
        return False, "hold_until_month", dict(current)
    return False, "hold_until_month", dict(current)


def trade_cost(current: Mapping[str, float], target: Mapping[str, float]) -> float:
    return TRADE_COST * turnover(current, target)


def metrics(equity: pd.Series) -> dict[str, float]:
    series = equity.dropna().astype(float)
    if len(series) < 2 or series.iloc[0] <= 0 or series.iloc[-1] <= 0:
        return {"cagr": 0.0, "vol": 0.0, "maxDrawdown": 0.0, "sharpe": 0.0, "totalReturn": 0.0}
    years = max((series.index[-1] - series.index[0]).days / 365.25, 1 / 365.25)
    total = float(series.iloc[-1] / series.iloc[0] - 1.0)
    cagr = float((series.iloc[-1] / series.iloc[0]) ** (1.0 / years) - 1.0)
    rets = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    vol = float(rets.std() * math.sqrt(252)) if len(rets) else 0.0
    peak = series.cummax()
    drawdown = float((series / peak - 1.0).min())
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if len(rets) and rets.std() > 0 else 0.0
    return {
        "cagr": cagr,
        "vol": vol,
        "maxDrawdown": drawdown,
        "sharpe": sharpe,
        "totalReturn": total,
    }


def shock_reasons(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    held: Mapping[str, float],
    vix_level: float | None,
    risk_off: bool,
) -> list[str]:
    """Reasons visible at `date`. Uses only closes through that session."""
    reasons: list[str] = []
    history = prices.loc[:date]
    if risk_off and vix_level is not None and math.isfinite(vix_level):
        reasons.append(f"VIX is {vix_level:.1f}, so risk stays capped")
    if len(history) < 6:
        return reasons
    window = history.tail(6)
    five = window.iloc[-1] / window.iloc[0] - 1.0
    one = window.iloc[-1] / window.iloc[-2] - 1.0
    flagged: set[str] = set()
    for ticker, weight in held.items():
        if ticker == CASH or weight < 0.05 or ticker not in one.index:
            continue
        if pd.notna(five[ticker]) and float(five[ticker]) <= SHOCK_FIVE_DAY:
            reasons.append(f"{ticker} is down {float(five[ticker]):.1%} over five sessions")
            flagged.add(ticker)
        if pd.notna(one[ticker]) and float(one[ticker]) <= SHOCK_ONE_DAY:
            reasons.append(f"{ticker} fell {float(one[ticker]):.1%} in the last session")
            flagged.add(ticker)
    if "SPY" not in flagged and "SPY" in one.index and pd.notna(one["SPY"]) and float(one["SPY"]) <= SHOCK_ONE_DAY:
        reasons.append(f"S&P 500 fell {float(one['SPY']):.1%} in the last session")
    return reasons
