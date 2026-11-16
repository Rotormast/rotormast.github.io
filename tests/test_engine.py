import unittest

import numpy as np
import pandas as pd

from strategy.engine import run_desk
from strategy.memo import plain_text
from strategy.universe import CASH


def _market(n=760, seed=4):
    index = pd.bdate_range("2012-01-02", periods=n)
    rng = np.random.default_rng(seed)
    def path(drift, vol, start=100.0):
        return start * np.exp(np.cumsum(drift + vol * rng.normal(size=n)))
    prices = pd.DataFrame(
        {
            "SPY": path(0.00035, 0.009),
            "QQQ": path(0.00045, 0.011),
            "IWM": path(0.0002, 0.012),
            "XLK": path(0.0004, 0.012),
            "XLP": path(0.00015, 0.007),
            CASH: path(0.00007, 0.0002),
        },
        index=index,
    )
    macro = pd.DataFrame(
        {
            "VIX": np.clip(16 + rng.normal(0, 2, n), 10, 28),
            "TLT": path(0.00005, 0.006),
            "SHY": path(0.00004, 0.001),
            "HYG": path(0.00012, 0.004),
            "LQD": path(0.00008, 0.003),
            "UUP": path(0.0, 0.003),
        },
        index=index,
    )
    return prices, macro


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = run_desk(*_market(), model_dir=None)

    def test_instruction_is_a_complete_book(self):
        weights = [row["weight"] for row in self.state["instruction"]]
        self.assertAlmostEqual(sum(weights), 1.0, places=6)
        self.assertTrue(all(weight >= -1e-9 for weight in weights))
        self.assertEqual(self.state["phase"], "ready")

    def test_trades_do_not_use_future_labels(self):
        index = pd.bdate_range("2012-01-02", periods=760)
        self.assertGreaterEqual(len(self.state["trades"]), 1)
        for trade in self.state["trades"]:
            feature = pd.Timestamp(trade["featureDate"])
            cutoff = pd.Timestamp(trade["labelCutoff"])
            self.assertLessEqual(feature, pd.Timestamp(trade["date"]))
            self.assertGreaterEqual(int(index.get_loc(feature)) - int(index.get_loc(cutoff)), 21)

    def test_objective_is_explicit(self):
        objective = self.state["objective"]
        self.assertIn("beatEveryIndex", objective)
        self.assertIsInstance(objective["beatEveryIndex"], bool)
        self.assertLess(objective["selectionEnd"], objective["windowStart"])

    def test_memo_names_the_advisor_and_the_weights(self):
        text = plain_text(self.state)
        self.assertIn("Merrill Lynch", text)
        self.assertIn("Treasury bills (BIL)", text)
        self.assertIn("not an order", text)


if __name__ == "__main__":
    unittest.main()
