import json
import subprocess
import tempfile
import unittest
from datetime import date
from pathlib import Path

import tests  # noqa: F401  (installs the stand-ins)
from src.JBGModelPrices import (call_cost, format_cost, price_for, read_price_table,
                                validate_price_table)
from src.JBGBuildInfo import read_build_info


class PriceTableTest(unittest.TestCase):
    def setUp(self):
        self.table = read_price_table()

    def table_with(self, **changes):
        data = json.loads(Path("policy/model_prices.json").read_text(encoding="utf-8"))
        data.update(changes)
        path = Path(tempfile.mkdtemp()) / "prices.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return read_price_table(path)

    def test_menu_models_all_have_prices(self):
        self.assertFalse(self.table["errors"])
        self.assertIn(self.table["default"], self.table["menu"])
        for model in self.table["menu"]:
            self.assertIsNotNone(price_for(model, self.table), model)

    def test_name_matching(self):
        self.assertEqual(price_for("gpt-5.2-2025-12-11", self.table), price_for("gpt-5.2", self.table))
        self.assertEqual(price_for("gpt-5.4-mini", self.table)["in"], 0.75)
        self.assertIsNone(price_for("gpt-5.4-turbo", self.table))

    def test_cost_formula_with_cache(self):
        # The run confirmed against OpenAI's prices: 2 281 in, 1 653 out.
        self.assertAlmostEqual(call_cost("gpt-5.2", 2281, 0, 1653, self.table), 0.02713, places=5)
        cost = call_cost("gpt-5.2", 1_000_000, 880_000, 100_000, self.table)
        self.assertAlmostEqual(cost, (120_000 * 1.75 + 880_000 * 0.175 + 100_000 * 14) / 1e6)
        self.assertIsNone(call_cost("unknown", 1, 0, 1, self.table))

    def test_startup_warnings(self):
        today = date(2026, 10, 23)
        self.assertEqual(validate_price_table(self.table_with(_kontrollerad="2026-09-01"), today), [])
        self.assertTrue(any("dagar sedan" in w for w in
                            validate_price_table(self.table_with(_kontrollerad="2026-07-01"), today)))
        self.assertTrue(any("saknar pris" in w for w in validate_price_table(
            self.table_with(_kontrollerad="2026-10-01", _i_menyn=["gpt-5.2", "gpt-9"]), today)))

    def test_swedish_formatting(self):
        self.assertEqual([format_cost(v) for v in (0.4182, 12.5, 0.0031)], ["0,42", "12,50", "0,0031"])


class BuildInfoTest(unittest.TestCase):
    def test_deployed_and_local(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "VERSION").write_text("9.9.9\n")
            self.assertEqual(read_build_info(root)["label"], "Version 9.9.9 (lokal)")
            (root / "build_info.json").write_text(json.dumps({
                "commit": "a1b2c3d4e5", "environment": "prod",
                "repository_url": "https://github.com/org/repo"}))
            info = read_build_info(root)
            self.assertEqual(info["label"], "Version 9.9.9 · a1b2c3d (prod)")
            self.assertEqual(info["commit_url"], "https://github.com/org/repo/commit/a1b2c3d4e5")
