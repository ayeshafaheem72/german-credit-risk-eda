"""Regression tests run in CI on every push: the exported model must load, score, and keep its quality."""
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "serving"))
from train_and_export import load_data, main as train  # noqa: E402


def setup_module():
    train()


def test_model_scores_a_valid_probability():
    model = joblib.load(ROOT / "serving" / "model.joblib")
    _, X_test, _, _ = load_data()
    p = model.predict_proba(X_test.iloc[[0]])[0, 1]
    assert 0.0 <= p <= 1.0


def test_recall_on_bad_risk_does_not_regress():
    meta = json.loads((ROOT / "serving" / "model_meta.json").read_text())
    assert meta["recall_bad"] >= 0.75, "recall on bad-risk applicants dropped below 75%"


def test_business_cost_beats_majority_baseline():
    meta = json.loads((ROOT / "serving" / "model_meta.json").read_text())
    assert meta["business_cost"] < 300  # always-approve baseline cost on the same test split


def test_single_request_latency_under_200ms():
    model = joblib.load(ROOT / "serving" / "model.joblib")
    row = pd.read_csv(ROOT / "data" / "german_credit_clean.csv").drop(columns="target").iloc[[0]]
    model.predict_proba(row)  # warm-up
    t = time.perf_counter()
    model.predict_proba(row)
    assert (time.perf_counter() - t) * 1000 < 200
