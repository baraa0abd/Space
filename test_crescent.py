"""Run: python -m unittest test_crescent -v"""
from datetime import date, datetime
import unittest
from unittest.mock import patch
import pandas as pd
import crescent


class CrescentTests(unittest.TestCase):
    def test_ramadan_and_shawwal_conjunctions(self):
        for event, expected in [("ramadan", date(2024, 3, 10)), ("fitr", date(2024, 4, 8))]:
            _, evening = crescent.campaign(event, 1445)
            self.assertEqual(evening, expected)

    def test_future_does_not_use_weather_defaults(self):
        with patch("skyq.weather") as wx, patch("skyq.hilal_evening", return_value={"zone": "A"}):
            rows = crescent.plan(36.34, 43.13, date(2027, 2, 7), today=date(2026, 9, 26))
        wx.assert_not_called()
        self.assertTrue(rows.local_zone.isna().all())

    def test_weather_failure_preserves_geometry_only(self):
        evening = date(2026, 9, 26)
        with patch("skyq.hilal_evening", return_value={"zone": "B", "best_time": datetime(2026, 9, 26, 18)}), \
             patch("skyq.weather", return_value=(pd.DataFrame(), ["using climatology"])):
            rows = crescent.plan(36.34, 43.13, evening, nights=1, today=evening)
        self.assertEqual(rows.iloc[0].zone, "B")
        self.assertIsNone(rows.iloc[0].local_zone)

    def test_main_page_loads(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py").run(timeout=30)
        self.assertFalse(at.exception)
        self.assertIn("رمضان", at.title[0].value)

    def test_calculation_renders_without_official_date(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("ramadan.py").run(timeout=30)
        at.number_input[0].set_value(1445)
        at.button[0].click().run(timeout=120)
        self.assertFalse(at.exception)
        self.assertTrue(at.session_state["crescent_result"]["frame"].shape[0] == 3)
        self.assertEqual(len(at.metric), 12)


if __name__ == "__main__":
    unittest.main()
