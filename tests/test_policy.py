import unittest

import pandas as pd

from strategy.policy import allocate, apply_shock, decide, metrics, shock_reasons, trade_cost, turnover
from strategy.universe import CASH


class PolicyTests(unittest.TestCase):
    def test_negative_forecasts_hold_cash(self):
        weights = allocate({"SPY": -0.01, "QQQ": -0.02, CASH: 0.0}, {"SPY": 0.16, "QQQ": 0.22}, 1.0)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights[CASH], 1.0)
        self.assertTrue(all(value >= -1e-12 for value in weights.values()))

    def test_cap_and_budget(self):
        weights = allocate({"SPY": 0.04, "QQQ": 0.01}, {"SPY": 0.15, "QQQ": 0.2}, risk_aversion=0.25, cap=0.70)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertLessEqual(weights["SPY"], 0.70 + 1e-9)
        self.assertGreater(weights["SPY"], weights["QQQ"])
        self.assertGreater(weights[CASH], 0.0)

    def test_shock_only_cuts_risk(self):
        current = {"SPY": 0.8, "QQQ": 0.2, CASH: 0.0}
        shocked = apply_shock(current, gross_cap=0.40)
        self.assertAlmostEqual(sum(shocked.values()), 1.0)
        self.assertLessEqual(shocked["SPY"] + shocked["QQQ"], 0.40 + 1e-9)
        self.assertAlmostEqual(shocked[CASH], 0.60, places=6)
        already = apply_shock({"SPY": 0.2, CASH: 0.8}, gross_cap=0.40)
        self.assertAlmostEqual(already["SPY"], 0.2, places=6)

    def test_weekly_lock_and_shock_exception(self):
        today = pd.Timestamp("2026-11-13")
        last = pd.Timestamp("2026-11-10")
        current = {"SPY": 0.5, CASH: 0.5}
        target = {"SPY": 0.2, CASH: 0.8}
        trade, reason, _weights = decide(today, last, "monthly", current, target, shock=False)
        self.assertFalse(trade)
        self.assertEqual(reason, "hold_weekly_lock")
        trade, reason, weights = decide(today, last, "monthly", current, target, shock=True, shock_entry=True)
        self.assertTrue(trade)
        self.assertEqual(reason, "shock")
        self.assertAlmostEqual(weights["SPY"], 0.2)
        trade, reason, _weights = decide(today, last, "shock", current, target, shock=True, shock_entry=False)
        self.assertFalse(trade)
        self.assertEqual(reason, "hold_shock")

    def test_monthly_trade_when_the_book_has_moved(self):
        trade, reason, _weights = decide(
            pd.Timestamp("2026-12-01"),
            pd.Timestamp("2026-11-02"),
            "monthly",
            {"SPY": 0.5, CASH: 0.5},
            {"SPY": 0.8, CASH: 0.2},
            shock=False,
        )
        self.assertTrue(trade)
        self.assertEqual(reason, "monthly")

    def test_cost_is_five_basis_points_per_side(self):
        cost = trade_cost({CASH: 1.0}, {"SPY": 1.0, CASH: 0.0})
        self.assertAlmostEqual(cost, 0.001)
        self.assertAlmostEqual(turnover({CASH: 1.0}, {"SPY": 1.0}), 2.0)

    def test_known_cagr(self):
        index = pd.bdate_range("2020-01-01", periods=253)
        equity = pd.Series(1.0, index=index)
        equity.iloc[-1] = 2.0
        # About one year of business days. Allow a wide bound around a doubling.
        self.assertGreater(metrics(equity)["cagr"], 0.9)
        self.assertLess(metrics(equity)["cagr"], 1.2)

    def test_shock_reason_uses_only_the_window(self):
        index = pd.bdate_range("2026-10-01", periods=8)
        prices = pd.DataFrame({"SPY": [100, 100, 100, 102, 101, 100, 96, 84], CASH: 100.0}, index=index)
        reasons = shock_reasons(prices, index[-1], {"SPY": 0.5, CASH: 0.5}, 18, False)
        self.assertTrue(any("SPY" in reason or "S&P 500" in reason for reason in reasons))
        flat = pd.DataFrame({"SPY": [100.0] * 8, CASH: 100.0}, index=index)
        self.assertEqual(shock_reasons(flat, index[-1], {"SPY": 0.5, CASH: 0.5}, 18, False), [])


if __name__ == "__main__":
    unittest.main()
