import unittest

import numpy as np
import pandas as pd

from strategy.factors import build_panel


def _market(n=420, seed=2):
    index = pd.bdate_range("2016-01-04", periods=n)
    rng = np.random.default_rng(seed)
    def path(drift, vol, start=100.0):
        return start * np.exp(np.cumsum(drift + vol * rng.normal(size=n)))
    prices = pd.DataFrame(
        {
            "SPY": path(0.0003, 0.01),
            "QQQ": path(0.0004, 0.012),
            "BIL": path(0.00007, 0.0002, start=50.0),
        },
        index=index,
    )
    macro = pd.DataFrame(
        {
            "VIX": 16 + np.abs(rng.normal(0, 1, n)),
            "TLT": path(0.0001, 0.005),
            "SHY": path(0.00005, 0.001),
            "HYG": path(0.0001, 0.004),
            "LQD": path(0.00008, 0.003),
            "UUP": path(0.0, 0.004),
        },
        index=index,
    )
    return prices, macro


class FactorTests(unittest.TestCase):
    def test_future_prices_do_not_change_past_features(self):
        prices, macro = _market()
        panel = build_panel(prices, macro)
        mid = prices.index[300]
        features = panel.features
        before = panel.frame[(panel.frame["date"] == mid) & (panel.frame["ticker"] == "SPY")][features].iloc[0].to_numpy()
        changed = prices.copy()
        changed.iloc[301:] = changed.iloc[301:] * 1.5
        after = build_panel(changed, macro)
        later = after.frame[(after.frame["date"] == mid) & (after.frame["ticker"] == "SPY")][features].iloc[0].to_numpy()
        self.assertTrue(np.allclose(before, later, equal_nan=True))
        label_before = panel.frame[(panel.frame["date"] == mid) & (panel.frame["ticker"] == "SPY")]["fwd_excess"].iloc[0]
        label_after = after.frame[(after.frame["date"] == mid) & (after.frame["ticker"] == "SPY")]["fwd_excess"].iloc[0]
        self.assertFalse(np.isclose(label_before, label_after))


if __name__ == "__main__":
    unittest.main()
