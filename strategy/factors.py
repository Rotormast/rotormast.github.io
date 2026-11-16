"""Factor panel. Every feature at date t uses closes through t. The label looks ahead 21 sessions and is never a feature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy.universe import CASH, RISKY

FEATURES = (
    "mom_21",
    "mom_63",
    "mom_126",
    "mom_252",
    "vol_21",
    "vol_63",
    "drawdown",
    "ma_gap",
    "rs_63",
    "vix",
    "vix_chg_21",
    "term_63",
    "credit_63",
    "usd_63",
)


@dataclass
class Panel:
    frame: pd.DataFrame
    features: list[str]
    prices: pd.DataFrame
    vix: pd.Series


def _series(macro: pd.DataFrame, column: str, index: pd.DatetimeIndex) -> pd.Series:
    if column not in macro.columns:
        return pd.Series(np.nan, index=index, dtype=float)
    return macro[column].reindex(index).ffill(limit=5).astype(float)


def build_panel(prices: pd.DataFrame, macro: pd.DataFrame) -> Panel:
    prices = prices.sort_index().astype(float)
    prices = prices[~prices.index.duplicated(keep="last")]
    risky = [ticker for ticker in RISKY if ticker in prices.columns]
    rets = prices.pct_change()
    momentum = {lag: prices / prices.shift(lag) - 1.0 for lag in (21, 63, 126, 252)}
    vol_21 = rets.rolling(21).std() * np.sqrt(252)
    vol_63 = rets.rolling(63).std() * np.sqrt(252)
    drawdown = prices / prices.rolling(252, min_periods=63).max() - 1.0
    ma_gap = prices / prices.rolling(200, min_periods=100).mean() - 1.0
    cross = momentum[63][risky]
    rs_63 = cross.sub(cross.median(axis=1), axis=0)

    index = prices.index
    aligned = macro.reindex(index).ffill(limit=5) if len(macro.columns) else macro.reindex(index)
    vix = _series(aligned, "VIX", index)
    vix_chg = vix.pct_change(21)

    def macro_return(column: str, lag: int = 63) -> pd.Series:
        series = _series(aligned, column, index)
        return series.pct_change(lag)

    term = macro_return("TLT") - macro_return("SHY")
    credit = macro_return("HYG") - macro_return("LQD")
    usd = macro_return("UUP")
    bil_fwd = prices[CASH].shift(-21) / prices[CASH] - 1.0 if CASH in prices.columns else 0.0

    shared = {
        "vix": vix.to_numpy(dtype=float),
        "vix_chg_21": vix_chg.to_numpy(dtype=float),
        "term_63": term.to_numpy(dtype=float),
        "credit_63": credit.to_numpy(dtype=float),
        "usd_63": usd.to_numpy(dtype=float),
    }
    frames = []
    for ticker in risky:
        excess = prices[ticker].shift(-21) / prices[ticker] - 1.0 - bil_fwd
        frame = pd.DataFrame(
            {
                "date": index,
                "ticker": ticker,
                "mom_21": momentum[21][ticker].to_numpy(dtype=float),
                "mom_63": momentum[63][ticker].to_numpy(dtype=float),
                "mom_126": momentum[126][ticker].to_numpy(dtype=float),
                "mom_252": momentum[252][ticker].to_numpy(dtype=float),
                "vol_21": vol_21[ticker].to_numpy(dtype=float),
                "vol_63": vol_63[ticker].to_numpy(dtype=float),
                "drawdown": drawdown[ticker].to_numpy(dtype=float),
                "ma_gap": ma_gap[ticker].to_numpy(dtype=float),
                "rs_63": rs_63[ticker].to_numpy(dtype=float) if ticker in rs_63.columns else np.nan,
                "fwd_excess": excess.to_numpy(dtype=float),
                **shared,
            }
        )
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    features = [name for name in FEATURES if panel[name].notna().mean() > 0.5]
    return Panel(frame=panel, features=features, prices=prices, vix=vix)
