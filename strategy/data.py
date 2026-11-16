"""Daily closes for the desk. Cached on disk so a refresh does not redownload the full history."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from strategy.universe import ALL_TICKERS, MACRO_TICKERS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
MAX_AGE = timedelta(hours=6)
AST = timezone(timedelta(hours=-4))


def _extract_close(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        if "Close" in frame.columns.get_level_values(0):
            close = frame["Close"].copy()
        elif "Close" in frame.columns.get_level_values(1):
            close = frame.xs("Close", axis=1, level=1).copy()
        else:
            raise RuntimeError("Price download did not include closes.")
    else:
        close = frame.copy()
    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close.index = close.index.normalize()
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")]
    return close.apply(pd.to_numeric, errors="coerce")


def _download() -> pd.DataFrame:
    import yfinance as yf

    symbols = list(dict.fromkeys([*ALL_TICKERS, *MACRO_TICKERS.keys()]))
    frame = yf.download(
        symbols,
        start="2004-01-01",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if frame is None or len(frame) == 0:
        raise RuntimeError("The price download was empty.")
    return _extract_close(frame)


def _split(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro_columns = {}
    for raw, name in MACRO_TICKERS.items():
        if raw in close.columns:
            macro_columns[name] = close[raw]
    macro = pd.DataFrame(macro_columns, index=close.index)
    prices = close[[ticker for ticker in ALL_TICKERS if ticker in close.columns]].copy()
    prices = prices.dropna(how="all")
    return prices, macro


def _read_cache() -> tuple[pd.DataFrame, pd.DataFrame, datetime] | None:
    prices_path = CACHE / "prices.csv"
    macro_path = CACHE / "macro.csv"
    meta_path = CACHE / "meta.json"
    if not (prices_path.exists() and macro_path.exists() and meta_path.exists()):
        return None
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    meta = json.loads(meta_path.read_text())
    saved = datetime.fromisoformat(meta["savedAt"])
    return prices, macro, saved


def _write_cache(prices: pd.DataFrame, macro: pd.DataFrame) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    prices.to_csv(CACHE / "prices.csv")
    macro.to_csv(CACHE / "macro.csv")
    (CACHE / "meta.json").write_text(
        json.dumps({"savedAt": datetime.now(AST).isoformat(timespec="minutes")})
    )


def load_market(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Return prices, macro, and an optional warning. Uses the cache when it is fresh."""
    cached = _read_cache()
    if cached and not refresh:
        prices, macro, saved = cached
        age = datetime.now(saved.tzinfo) - saved
        if age <= MAX_AGE:
            return prices, macro, None
    try:
        prices, macro = _split(_download())
    except Exception as error:
        if cached:
            prices, macro, saved = cached
            return prices, macro, f"Using cached closes from {saved.isoformat(timespec='minutes')}. Refresh failed: {error}"
        raise
    if prices.empty or "SPY" not in prices.columns or "BIL" not in prices.columns:
        raise RuntimeError("The download did not include the S&P 500 and Treasury bill series.")
    _write_cache(prices, macro)
    return prices, macro, None
