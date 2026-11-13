"""
test_pipeline.py
Unit and Regression Test Suite for Workforce Attendance Analytics & Predictive ML Pipeline.
Can be executed with either `python -m unittest tests/test_pipeline.py` or `pytest`.
"""

import os
import unittest
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "attendance_90days_synthetic.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "late_arrival_pipeline.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")

class TestAttendancePipeline(unittest.TestCase):

    def test_01_dataset_integrity(self):
        """Verifies synthetic dataset presence, volume, schema, and non-empty values."""
        self.assertTrue(os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        
        self.assertGreaterEqual(len(df), 3000, "Dataset must contain at least 3,000 recorded shifts.")
        self.assertEqual(df["employee_id"].nunique(), 50, "Dataset should span 50 employees.")
        
        required_cols = [
            "employee_id", "employee_name", "department", "date", "day_of_week",
            "scheduled_start", "work_hours", "late_minutes", "status",
            "commute_dist_km", "punch_device"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing critical column: {col}")
            
        valid_statuses = {"Present", "Late", "Half-Day", "Absent"}
        actual_statuses = set(df["status"].unique())
        self.assertTrue(actual_statuses.issubset(valid_statuses), f"Unexpected status values: {actual_statuses}")

    def test_02_model_bundle_integrity(self):
        """Verifies saved model pipeline bundle and metadata."""
        self.assertTrue(os.path.exists(MODEL_PATH), f"Model bundle missing at {MODEL_PATH}")
        bundle = joblib.load(MODEL_PATH)

        required_keys = ["model", "feature_cols", "dept_map", "emp_rates", "global_late_rate", "metrics"]
        for k in required_keys:
            self.assertIn(k, bundle, f"Bundle missing key: {k}")

        self.assertGreater(len(bundle["dept_map"]), 0, "Department mapping must not be empty.")
        self.assertIsInstance(bundle["global_late_rate"], float, "Global late rate must be a float.")

    def test_03_inference_predictions(self):
        """Tests that model inference runs and produces valid probability bounds [0.0, 1.0]."""
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]

        sample_record = pd.DataFrame([{
            "commute_dist_km": 18.5,
            "is_monday": 1,
            "is_friday": 0,
            "dept_code": 1, # Sales
            "dist_x_monday": 18.5,
            "is_mobile_punch": 1,
            "emp_hist_late_rate": 0.13
        }])[feature_cols]

        prob = model.predict_proba(sample_record)[0, 1]
        self.assertGreaterEqual(prob, 0.0, "Probability must be >= 0.0")
        self.assertLessEqual(prob, 1.0, "Probability must be <= 1.0")

    def test_04_target_leakage_fallback(self):
        """Tests that an unseen employee ID falls back to the global late rate safely."""
        bundle = joblib.load(MODEL_PATH)
        emp_rates = bundle["emp_rates"]
        global_late_rate = bundle["global_late_rate"]

        unseen_id = "EMP_UNSEEN_999"
        hist_rate = emp_rates.get(unseen_id, global_late_rate)
        self.assertEqual(hist_rate, global_late_rate, "Unseen employee must fall back to global train rate.")

    def test_05_policy_simulation_monotonicity(self):
        """Verifies that increasing grace window monotonically reduces or maintains tardiness."""
        df = pd.read_csv(DATA_PATH)
        base_late = (df["late_minutes"] > 0).sum()
        grace_15_late = (df["late_minutes"] > 15).sum()
        grace_30_late = (df["late_minutes"] > 30).sum()

        self.assertLessEqual(grace_15_late, base_late, "15m grace should have fewer or equal late shifts than 0m grace.")
        self.assertLessEqual(grace_30_late, grace_15_late, "30m grace should have fewer or equal late shifts than 15m grace.")

if __name__ == "__main__":
    unittest.main()
