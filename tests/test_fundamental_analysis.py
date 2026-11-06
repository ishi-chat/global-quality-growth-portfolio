import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).parents[1] / "fundamental_analysis.ipynb"


def _load_calculate_fundamentals():
    notebook = json.loads(NOTEBOOK.read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def calculate_fundamentals" in "".join(cell.get("source", []))
    )

    def get_metric(statement, possible_names):
        for name in possible_names:
            if name in statement.index:
                return pd.to_numeric(statement.loc[name], errors="coerce")
        return pd.Series(np.nan, index=statement.columns, dtype=float)

    statements = {}

    def load_financials(ticker):
        return statements[ticker]

    namespace = {
        "get_metric": get_metric,
        "load_financials": load_financials,
        "np": np,
        "pd": pd,
    }
    exec(source, namespace)
    return namespace["calculate_fundamentals"], statements


class FundamentalAnalysisTests(unittest.TestCase):
    def test_free_cash_flow_treats_capex_as_an_outflow_regardless_of_sign(self):
        calculate_fundamentals, statements = _load_calculate_fundamentals()
        dates = pd.to_datetime(["2023-12-31", "2024-12-31"])

        income = pd.DataFrame(
            {
                dates[0]: [1000, 200, 150, 180, 18],
                dates[1]: [1200, 240, 180, 220, 22],
            },
            index=[
                "Total Revenue",
                "Operating Income",
                "Net Income",
                "Pretax Income",
                "Tax Provision",
            ],
        )
        balance = pd.DataFrame(
            {
                dates[0]: [100, 500, 300],
                dates[1]: [120, 520, 320],
            },
            index=["Total Debt", "Cash And Cash Equivalents", "Stockholders Equity"],
        )

        for capex in (-20, 20):
            with self.subTest(capex=capex):
                cashflow = pd.DataFrame(
                    {dates[0]: [100, capex], dates[1]: [120, capex]},
                    index=["Operating Cash Flow", "Capital Expenditure"],
                )
                statements["TEST"] = (income, balance, cashflow)

                result = calculate_fundamentals("TEST")

                self.assertEqual(result.loc[dates[1], "FCF"], 100)


if __name__ == "__main__":
    unittest.main()
