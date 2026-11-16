"""Walk-forward allocation.

Factor identity is chosen on older data. Each later month refits coefficients
using only labels whose 21-session outcome is already known. The headline
track record starts after that factor choice.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from strategy.factors import Panel, build_panel
from strategy.policy import allocate, apply_shock, decide, drift, metrics, shock_reasons, trade_cost, turnover
from strategy.universe import BY_TICKER, CASH, RISK_GRID, VIX_ENTER, VIX_EXIT

AST = timezone(timedelta(hours=-4))
ACTION_LABELS = {
    "initial": "Set the book",
    "monthly": "Monthly reallocation",
    "weekly": "Weekly reallocation",
    "shock": "Real-time adjustment",
    "hold_weekly_lock": "Hold",
    "hold_until_month": "Hold",
    "hold_shock": "Hold",
}


def _month_starts(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    starts: list[pd.Timestamp] = []
    seen: set[tuple[int, int]] = set()
    for stamp in index:
        key = (int(stamp.year), int(stamp.month))
        if key in seen:
            continue
        seen.add(key)
        starts.append(pd.Timestamp(stamp))
    return starts


def _design(frame: pd.DataFrame, features: list[str], require_label: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    columns = list(features) + (["fwd_excess"] if require_label else [])
    subset = frame.dropna(subset=columns)
    if subset.empty:
        empty = np.zeros((0, len(features)))
        return subset, empty, (np.zeros(0) if require_label else None)
    matrix = subset[list(features)].to_numpy(dtype=float)
    target = subset["fwd_excess"].to_numpy(dtype=float) if require_label else None
    return subset, matrix, target


def _fit(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    _, matrix, target = _design(frame, features, True)
    if len(matrix) < 80 or len(features) == 0:
        return None
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    model = Ridge(alpha=1.0)
    model.fit((matrix - mean) / scale, target)
    return mean, scale, model.coef_.astype(float), float(model.intercept_)


def _predict_rows(frame: pd.DataFrame, features: list[str], mean, scale, coef, intercept) -> tuple[dict[str, float], dict[str, float]]:
    subset, matrix, _ = _design(frame, features, False)
    mu: dict[str, float] = {}
    vol: dict[str, float] = {}
    if len(matrix) == 0:
        return mu, vol
    predicted = np.clip((matrix - mean) / scale @ coef + intercept, -0.5, 0.5)
    for ticker, value, realized_vol in zip(subset["ticker"], predicted, subset["vol_21"], strict=True):
        mu[str(ticker)] = float(value)
        vol[str(ticker)] = float(realized_vol) if np.isfinite(realized_vol) else 0.2
    return mu, vol


def _portfolio_excess(weights: dict[str, float], realized: dict[str, float]) -> float:
    return float(sum(weight * realized.get(ticker, 0.0) for ticker, weight in weights.items() if ticker != CASH))


def _score_window(
    panel: pd.DataFrame,
    features: list[str],
    index: pd.DatetimeIndex,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> tuple[float, float]:
    """Fit before val_start, score month-end outcomes through val_end. Returns score and risk."""
    train = panel[(panel["date"] < val_start) & panel["fwd_excess"].notna()]
    fitted = _fit(train, features)
    if fitted is None:
        return -1.0, 1.0
    mean, scale, coef, intercept = fitted
    val_dates = [stamp for stamp in _month_starts(index) if val_start <= stamp <= val_end]
    grouped = {pd.Timestamp(stamp): frame for stamp, frame in panel.groupby("date", sort=False)}
    best_risk = 1.0
    best_score = -1e9
    for risk in RISK_GRID:
        realized: list[float] = []
        for stamp in val_dates:
            rows = grouped.get(pd.Timestamp(stamp))
            if rows is None:
                continue
            labeled = rows.dropna(subset=["fwd_excess"])
            if labeled.empty:
                continue
            mu, vol = _predict_rows(rows, features, mean, scale, coef, intercept)
            if not mu:
                continue
            weights = allocate(mu, vol, risk)
            outcomes = {str(ticker): float(value) for ticker, value in zip(labeled["ticker"], labeled["fwd_excess"], strict=True)}
            realized.append(_portfolio_excess(weights, outcomes))
        if len(realized) < 3:
            continue
        arr = np.asarray(realized, dtype=float)
        score = float(arr.mean() - 0.15 * arr.std())
        if score > best_score:
            best_score = score
            best_risk = risk
    if best_score < -1e8:
        return -1.0, 1.0
    return best_score, best_risk


def select_features(
    panel: pd.DataFrame,
    features: list[str],
    index: pd.DatetimeIndex,
    selection_end: pd.Timestamp,
) -> tuple[list[str], list[str], float, float]:
    val_start = pd.Timestamp(selection_end) - pd.DateOffset(months=24)
    base, base_risk = _score_window(panel, features, index, val_start, pd.Timestamp(selection_end))
    kept: list[str] = []
    rejected: list[str] = []
    for feature in features:
        trial = [name for name in features if name != feature]
        if len(trial) < 4:
            kept.append(feature)
            continue
        score, _ = _score_window(panel, trial, index, val_start, pd.Timestamp(selection_end))
        if score > base + 1e-4:
            rejected.append(feature)
        else:
            kept.append(feature)
    if len(kept) < 4:
        return list(features), [], base, base_risk
    reduced, reduced_risk = _score_window(panel, kept, index, val_start, pd.Timestamp(selection_end))
    if reduced + 1e-4 < base:
        return list(features), [], base, base_risk
    return kept, rejected, reduced, reduced_risk


def choose_feature_set(
    panel: pd.DataFrame,
    features: list[str],
    index: pd.DatetimeIndex,
    selection_end: pd.Timestamp,
    incumbent: dict | None,
) -> dict:
    chosen, rejected, score, risk = select_features(panel, features, index, selection_end)
    action = "initial"
    if incumbent and incumbent.get("features"):
        old_features = [name for name in incumbent["features"] if name in features]
        if len(old_features) >= 4:
            old_score, old_risk = _score_window(
                panel, old_features, index, pd.Timestamp(selection_end) - pd.DateOffset(months=24), pd.Timestamp(selection_end)
            )
            if old_score >= score:
                chosen, rejected, score, risk = old_features, [name for name in features if name not in old_features], old_score, old_risk
                action = "refit"
            else:
                action = "promote"
        else:
            action = "promote"
    return {
        "features": chosen,
        "rejected": rejected,
        "score": score,
        "risk": risk,
        "action": action,
        "selectionEnd": pd.Timestamp(selection_end).date().isoformat(),
    }


def _weights_at(panel: pd.DataFrame, features: list[str], index: pd.DatetimeIndex, date: pd.Timestamp) -> dict | None:
    loc = int(index.get_loc(pd.Timestamp(date)))
    cutoff_loc = loc - 21
    if cutoff_loc < 252:
        return None
    cutoff = pd.Timestamp(index[cutoff_loc])
    train = panel[(panel["date"] <= cutoff) & panel["fwd_excess"].notna()]
    # Risk was chosen with the feature set; use the stored grid default if the
    # caller passes it via columns. The simulation passes risk separately.
    fitted = _fit(train, features)
    if fitted is None:
        return None
    rows = panel.loc[panel["date"] == pd.Timestamp(date)]
    mu, vol = _predict_rows(rows, features, *fitted)
    return {
        "mu": mu,
        "vol": vol,
        "fit": fitted,
        "cutoff": cutoff,
        "featureDate": pd.Timestamp(date),
    }


def _selection_end(index: pd.DatetimeIndex) -> pd.Timestamp | None:
    if len(index) < 273 + 40:
        return None
    last_labeled = pd.Timestamp(index[len(index) - 1 - 21])
    span_days = (last_labeled - pd.Timestamp(index[252])).days
    oos_days = 730 if span_days > 365 * 6 else max(90, int(span_days * 0.25))
    return last_labeled - pd.Timedelta(days=oos_days)


def simulate(panel: Panel, features: list[str], risk: float) -> dict:
    index = pd.DatetimeIndex(panel.prices.index)
    prices = panel.prices
    targets: dict[pd.Timestamp, dict] = {}
    for stamp in _month_starts(index):
        built = _weights_at(panel.frame, features, index, stamp)
        if built is None or not built["mu"]:
            continue
        targets[pd.Timestamp(stamp)] = built
    if not targets:
        raise RuntimeError("Not enough history to fit the allocation model.")
    last = pd.Timestamp(index[-1])
    if last not in targets:
        built = _weights_at(panel.frame, features, index, last)
        if built and built["mu"]:
            targets[last] = built
    first = min(targets)
    dates = index[index.get_loc(first) :]
    tickers = list(prices.columns)
    held = {ticker: (1.0 if ticker == CASH else 0.0) for ticker in tickers}
    last_trade: pd.Timestamp | None = None
    last_reason: str | None = None
    equity = 1.0
    curve: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []
    risk_off = False
    was_active = False
    current: dict | None = None
    pre_trade = dict(held)
    action = "hold_until_month"
    reasons: list[str] = []
    vix = panel.vix.reindex(index)

    for position, date in enumerate(dates):
        date = pd.Timestamp(date)
        if position > 0:
            prev_date = pd.Timestamp(dates[position - 1])
            change = {}
            for ticker in tickers:
                before = prices.at[prev_date, ticker] if ticker in prices.columns else np.nan
                after = prices.at[date, ticker] if ticker in prices.columns else np.nan
                if pd.notna(before) and pd.notna(after) and before > 0:
                    change[ticker] = float(after / before - 1.0)
                else:
                    change[ticker] = 0.0
            port = sum(held.get(ticker, 0.0) * change.get(ticker, 0.0) for ticker in held)
            equity *= 1.0 + port
            held = drift(held, change)

        if date in targets:
            current = targets[date]
        if current is None:
            curve.append((date, equity))
            continue

        level = vix.get(date, np.nan) if date in vix.index else np.nan
        try:
            level_value = float(level)
        except (TypeError, ValueError):
            level_value = float("nan")
        if np.isfinite(level_value):
            if risk_off and level_value < VIX_EXIT:
                risk_off = False
            elif not risk_off and level_value >= VIX_ENTER:
                risk_off = True
        price_reasons = shock_reasons(prices, date, held, None, False)
        reasons = list(price_reasons)
        if risk_off and np.isfinite(level_value):
            reasons.append(f"VIX is {level_value:.1f}, so risk stays capped")
        active = risk_off or bool(price_reasons)
        entry = active and not was_active
        desired = allocate(current["mu"], current["vol"], risk)
        if active:
            desired = apply_shock(desired)
        desired = _complete(desired, tickers)
        pre_trade = dict(held)
        trade, action, new_weights = decide(
            date, last_trade, last_reason, held, desired, shock=active, shock_entry=entry
        )
        was_active = active
        new_weights = _complete(new_weights, tickers)
        if trade:
            equity *= 1.0 - trade_cost(held, new_weights)
            trades.append(
                {
                    "date": date.date().isoformat(),
                    "reason": action,
                    "turnover": turnover(held, new_weights),
                    "featureDate": current["featureDate"].date().isoformat(),
                    "labelCutoff": current["cutoff"].date().isoformat(),
                }
            )
            held = new_weights
            last_trade = date
            last_reason = action
        curve.append((date, equity))

    series = pd.Series({stamp: value for stamp, value in curve}).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return {
        "equity": series,
        "trades": trades,
        "held": pre_trade,
        "instruction": held,
        "action": action,
        "reasons": reasons,
        "last": current,
        "asOf": pd.Timestamp(dates[-1]),
    }


def _complete(weights: dict[str, float], tickers: list[str]) -> dict[str, float]:
    out = {ticker: 0.0 for ticker in tickers}
    for ticker, weight in weights.items():
        if ticker in out:
            out[ticker] = float(weight)
        elif ticker == CASH:
            out[ticker] = float(weight)
    if CASH not in out:
        out[CASH] = 0.0
    total = sum(out.values())
    if total <= 0:
        return {ticker: (1.0 if ticker == CASH else 0.0) for ticker in out}
    return {ticker: value / total for ticker, value in out.items()}


def _slice_from(series: pd.Series, start: pd.Timestamp) -> pd.Series:
    part = series.loc[series.index >= pd.Timestamp(start)].dropna()
    if part.empty or part.iloc[0] <= 0:
        return part
    return part / float(part.iloc[0])


def _drivers(panel: pd.DataFrame, snapshot: dict | None, features: list[str], weights: dict[str, float]) -> list[dict]:
    if not snapshot:
        return []
    mean, scale, coef, _intercept = snapshot["fit"]
    feature_date = pd.Timestamp(snapshot["featureDate"])
    rows = panel.loc[panel["date"] == feature_date]
    ranked = sorted(
        ((ticker, weight) for ticker, weight in weights.items() if ticker != CASH and weight >= 0.02),
        key=lambda item: item[1],
        reverse=True,
    )
    notes = []
    for ticker, weight in ranked[:3]:
        match = rows.loc[rows["ticker"] == ticker]
        if match.empty or ticker not in snapshot["mu"]:
            continue
        values = match.iloc[0][features].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            continue
        effect = (values - mean) / scale * coef
        order = np.argsort(-np.abs(effect))[:3]
        notes.append(
            {
                "ticker": ticker,
                "name": BY_TICKER[ticker].name if ticker in BY_TICKER else ticker,
                "weight": float(weight),
                "expectedExcess": float(snapshot["mu"][ticker]),
                "factors": [{"name": features[int(pos)], "effect": float(effect[int(pos)])} for pos in order],
            }
        )
    return notes


def _chart(strategy: pd.Series, prices: pd.DataFrame, best: str, start: pd.Timestamp) -> dict:
    frame = pd.DataFrame({"strategy": _slice_from(strategy, start)})
    if "SPY" in prices.columns:
        frame["spy"] = _slice_from(_price_index(prices["SPY"], strategy.index), start)
    if best in prices.columns:
        frame["best"] = _slice_from(_price_index(prices[best], strategy.index), start)
    weekly = frame.resample("W-FRI").last().dropna(how="all")
    if len(weekly) > 520:
        weekly = weekly.iloc[:: max(1, len(weekly) // 520)]
    return {
        "dates": [stamp.date().isoformat() for stamp in weekly.index],
        "strategy": [None if pd.isna(value) else float(value) for value in weekly.get("strategy", [])],
        "spy": [None if pd.isna(value) else float(value) for value in weekly["spy"]] if "spy" in weekly else [],
        "best": [None if pd.isna(value) else float(value) for value in weekly["best"]] if "best" in weekly else [],
        "bestTicker": best,
    }


def _price_index(price: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    aligned = price.reindex(index).ffill()
    base = aligned.dropna()
    if base.empty or float(base.iloc[0]) <= 0:
        return aligned * np.nan
    return aligned / float(base.iloc[0])


def _scoreboard(strategy: pd.Series, prices: pd.DataFrame, start: pd.Timestamp) -> list[dict]:
    window = _slice_from(strategy, start)
    full = strategy / float(strategy.iloc[0])
    trail_start = window.index[-1] - pd.DateOffset(years=3) if len(window) else start
    rows = []
    for ticker in prices.columns:
        if ticker not in BY_TICKER:
            continue
        index_equity = _price_index(prices[ticker], strategy.index)
        matched = _slice_from(index_equity, start)
        common = pd.concat([window.rename("strategy"), matched.rename("index")], axis=1, join="inner").dropna()
        if len(common) < 2:
            continue
        strat = common["strategy"] / float(common["strategy"].iloc[0])
        bench = common["index"] / float(common["index"].iloc[0])
        left = metrics(strat)
        right = metrics(bench)
        recent_s = strat[strat.index >= trail_start]
        recent_b = bench[bench.index >= trail_start]
        gap3 = None
        bench3 = None
        if len(recent_s) > 40 and len(recent_b) > 40:
            gap3 = metrics(recent_s / float(recent_s.iloc[0]))["cagr"] - metrics(recent_b / float(recent_b.iloc[0]))["cagr"]
            bench3 = metrics(recent_b / float(recent_b.iloc[0]))["cagr"]
        info = BY_TICKER[ticker]
        rows.append(
            {
                "ticker": ticker,
                "name": info.name,
                "group": info.group,
                "cagr": right["cagr"],
                "vol": right["vol"],
                "maxDrawdown": right["maxDrawdown"],
                "totalReturn": right["totalReturn"],
                "strategyCagr": left["cagr"],
                "gap": left["cagr"] - right["cagr"],
                "beat": left["cagr"] > right["cagr"],
                "cagr3y": bench3,
                "gap3y": gap3,
            }
        )
    rows.sort(key=lambda row: (row["group"] != "broad", row["group"] == "cash", -row["cagr"]))
    return rows


def _books(weights: dict[str, float], other: dict[str, float] | None = None) -> list[dict]:
    rows = []
    for ticker, weight in weights.items():
        if ticker not in BY_TICKER:
            continue
        info = BY_TICKER[ticker]
        row = {
            "ticker": ticker,
            "name": info.name,
            "group": info.group,
            "weight": float(weight),
        }
        if other is not None:
            row["previous"] = float(other.get(ticker, 0.0))
            row["delta"] = float(weight) - float(other.get(ticker, 0.0))
        rows.append(row)
    rows.sort(key=lambda row: (row["group"] == "cash", -row["weight"]))
    return rows


def _industry_tape(prices: pd.DataFrame, as_of: pd.Timestamp, instruction: dict[str, float]) -> list[dict]:
    rows = []
    for ticker, info in BY_TICKER.items():
        if info.group != "industry" or ticker not in prices.columns:
            continue
        history = prices[ticker].loc[:as_of].dropna()
        def window(sessions: int) -> float | None:
            if len(history) <= sessions:
                return None
            return float(history.iloc[-1] / history.iloc[-1 - sessions] - 1.0)
        rows.append(
            {
                "ticker": ticker,
                "name": info.name,
                "weight": float(instruction.get(ticker, 0.0)),
                "ret5": window(5),
                "ret21": window(21),
                "ret63": window(63),
            }
        )
    rows.sort(key=lambda row: (row["ret5"] is None, -(row["ret5"] if row["ret5"] is not None else -999)))
    return rows


def _save_model(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    journal = path.parent / "journal.jsonl"
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({k: payload[k] for k in ("action", "selectionEnd", "score", "features", "rejected", "savedAt") if k in payload}) + "\n")


def load_incumbent(model_dir: Path | None) -> dict | None:
    if model_dir is None:
        return None
    path = model_dir / "current.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def run_desk(prices: pd.DataFrame, macro: pd.DataFrame, model_dir: Path | None = None) -> dict:
    panel = build_panel(prices, macro)
    index = pd.DatetimeIndex(panel.prices.index)
    selection_end = _selection_end(index)
    if selection_end is None:
        raise RuntimeError("Need more than about a year and a half of daily closes before the desk can train.")
    incumbent = load_incumbent(model_dir)
    choice = choose_feature_set(panel.frame, panel.features, index, selection_end, incumbent)
    result = simulate(panel, choice["features"], choice["risk"])
    as_of = result["asOf"]
    oos_start = pd.Timestamp(selection_end) + pd.Timedelta(days=1)
    board = _scoreboard(result["equity"], panel.prices, oos_start)
    contenders = [row for row in board if row["group"] != "cash"]
    best = max(contenders, key=lambda row: row["cagr"]) if contenders else None
    # Headline uses the shared window with the best index, so the gap is the same dates.
    strategy_cagr = best["strategyCagr"] if best else 0.0
    beat_all = bool(contenders) and all(row["beat"] for row in contenders)
    gap = best["gap"] if best else 0.0
    last = result["last"]
    coefficients = []
    if last is not None:
        _, _, coef, _ = last["fit"]
        coefficients = [
            {"name": feature, "value": float(coef[pos])} for pos, feature in enumerate(choice["features"])
        ]
        coefficients.sort(key=lambda item: abs(item["value"]), reverse=True)
    model_payload = {
        "action": choice["action"],
        "features": choice["features"],
        "rejected": choice["rejected"],
        "score": choice["score"],
        "riskAversion": choice["risk"],
        "selectionEnd": choice["selectionEnd"],
        "trainEnd": last["cutoff"].date().isoformat() if last else None,
        "savedAt": datetime.now(AST).isoformat(timespec="minutes"),
        "coefficients": coefficients,
    }
    if model_dir is not None:
        _save_model(model_dir / "current.json", model_payload)
    oos_equity = _slice_from(result["equity"], oos_start)
    window_start = oos_equity.index[0].date().isoformat() if len(oos_equity) else oos_start.date().isoformat()
    chart_best = best["ticker"] if best else "SPY"
    changes = []
    for row in _books(result["instruction"], result["held"]):
        if abs(row.get("delta", 0.0)) >= 0.01:
            changes.append(
                {
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "from": row["previous"],
                    "to": row["weight"],
                }
            )
    full_metrics = metrics(result["equity"] / float(result["equity"].iloc[0]))
    return {
        "phase": "ready",
        "asOf": as_of.date().isoformat(),
        "generatedAt": datetime.now(AST).isoformat(timespec="minutes"),
        "objective": {
            "beatEveryIndex": beat_all,
            "strategyCagr": strategy_cagr,
            "best": None
            if best is None
            else {"ticker": best["ticker"], "name": best["name"], "cagr": best["cagr"]},
            "gap": gap,
            "windowStart": window_start,
            "windowEnd": as_of.date().isoformat(),
            "selectionEnd": choice["selectionEnd"],
            "fullSampleCagr": full_metrics["cagr"],
            "fullSampleStart": result["equity"].index[0].date().isoformat(),
        },
        "action": {
            "code": result["action"],
            "label": ACTION_LABELS.get(result["action"], result["action"]),
            "trade": result["action"] in {"initial", "monthly", "weekly", "shock"},
            "reasons": result["reasons"],
        },
        "held": _books(result["held"]),
        "instruction": _books(result["instruction"], result["held"]),
        "changes": changes,
        "scoreboard": board,
        "industries": _industry_tape(panel.prices, as_of, result["instruction"]),
        "chart": _chart(result["equity"], panel.prices, chart_best, oos_start),
        "model": {
            **model_payload,
            "note": (
                "Factors were chosen with data through "
                f"{choice['selectionEnd']}. Each later month refits the coefficients "
                "on labels that are already known. The headline window is after that choice. "
                "Qwen does not set these weights."
            ),
        },
        "drivers": _drivers(panel.frame, last, choice["features"], result["instruction"]),
        "recentTrades": list(reversed(result["trades"][-8:])),
        "trades": result["trades"],
        "disclaimer": (
            "Personal research for Nate Hukill. Copy the allocation to the Merrill Lynch advisor, "
            "who decides whether to trade. This desk does not connect to Merrill Lynch and does not place orders. "
            "The track record is a walk-forward of daily closes with 5 basis points of cost per side. It is not a forecast."
        ),
    }
