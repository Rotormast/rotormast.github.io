"""Tradable indices and the macro series that inform them."""

from dataclasses import dataclass


CASH = "BIL"


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str
    group: str  # broad, industry, cash


UNIVERSE: tuple[Instrument, ...] = (
    Instrument("SPY", "S&P 500", "broad"),
    Instrument("QQQ", "Nasdaq-100", "broad"),
    Instrument("IWM", "Russell 2000", "broad"),
    Instrument("EFA", "MSCI EAFE", "broad"),
    Instrument("EEM", "MSCI Emerging Markets", "broad"),
    Instrument("XLK", "Technology", "industry"),
    Instrument("XLF", "Financials", "industry"),
    Instrument("XLE", "Energy", "industry"),
    Instrument("XLV", "Health care", "industry"),
    Instrument("XLI", "Industrials", "industry"),
    Instrument("XLY", "Consumer discretionary", "industry"),
    Instrument("XLP", "Consumer staples", "industry"),
    Instrument("XLU", "Utilities", "industry"),
    Instrument("XLB", "Materials", "industry"),
    Instrument("XLRE", "Real estate", "industry"),
    Instrument("XLC", "Communication services", "industry"),
    Instrument("BIL", "Treasury bills", "cash"),
)

MACRO_TICKERS: dict[str, str] = {
    "^VIX": "VIX",
    "TLT": "TLT",
    "SHY": "SHY",
    "HYG": "HYG",
    "LQD": "LQD",
    "UUP": "UUP",
}

BY_TICKER = {item.ticker: item for item in UNIVERSE}
RISKY = tuple(item.ticker for item in UNIVERSE if item.ticker != CASH)
ALL_TICKERS = tuple(item.ticker for item in UNIVERSE)

# One-way cost on dollars traded. A full switch of the book (sell one holding,
# buy another) turns over 2.0 of notional and costs 10 basis points.
TRADE_COST = 0.0005
MAX_WEIGHT = 0.70
TOP_HOLDINGS = 4
SHOCK_GROSS = 0.40
MIN_TURNOVER = 0.01
WEEKLY_DRIFT = 0.10
VIX_ENTER = 35.0
VIX_EXIT = 22.0
SHOCK_FIVE_DAY = -0.12
SHOCK_ONE_DAY = -0.06
RISK_GRID = (0.25, 0.5, 1.0, 2.0)
# Qwen3.8-27B is the current open-weight 27B model. This VM cannot load it.
QWEN_MODEL = "Qwen/Qwen3.8-27B"
QWEN_BASE_URL = "http://127.0.0.1:11434/v1"
