"""Gradio demo for the credit-risk model, with response-time and prediction logging.

Run:  python serving/train_and_export.py && python serving/app.py   (http://localhost:7860)
Every prediction is appended to serving/logs/predictions.csv with its latency, so model behaviour
and response time can be monitored over time.
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL = joblib.load(ROOT / "model.joblib")
META = json.loads((ROOT / "model_meta.json").read_text())
LOG = ROOT / "logs" / "predictions.csv"
LOG.parent.mkdir(exist_ok=True)

DATA = pd.read_csv(ROOT.parent / "data" / "german_credit_clean.csv")
CHOICES = {c: sorted(DATA[c].unique().tolist()) for c in DATA.columns if DATA[c].dtype == object}
NUMERIC = ["duration_months", "credit_amount", "installment_rate_pct", "present_residence_since",
           "age_years", "existing_credits_count", "num_dependents"]
COLUMNS = [c for c in DATA.columns if c != "target"]


def predict_one(row: dict) -> tuple[float, float]:
    """Return (probability of bad risk, latency in ms)."""
    start = time.perf_counter()
    proba = float(MODEL.predict_proba(pd.DataFrame([row])[COLUMNS])[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000
    new_file = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "probability_bad", "decision", "latency_ms"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), round(proba, 4),
                    "review" if proba >= 0.5 else "approve", round(latency_ms, 2)])
    return proba, latency_ms


def ui_predict(*values):
    row = dict(zip(COLUMNS, values))
    proba, latency = predict_one(row)
    decision = "Refer for manual review (high risk)" if proba >= 0.5 else "Low risk"
    return f"{proba:.1%}", decision, f"{latency:.1f} ms"


def build_inputs():
    example = DATA.iloc[0]
    inputs = []
    for c in COLUMNS:
        label = c.replace("_", " ")
        if c in NUMERIC:
            inputs.append(gr.Number(label=label, value=float(example[c])))
        else:
            inputs.append(gr.Dropdown(CHOICES[c], label=label, value=example[c]))
    return inputs


with gr.Blocks(title="Credit Risk Scoring Demo") as demo:
    gr.Markdown(
        "## Credit risk scoring demo\n"
        f"Model: {META['model']}. Test recall on bad risk: **{META['recall_bad']:.0%}**, "
        f"ROC-AUC {META['roc_auc']}. Trained on the public UCI German Credit dataset; for demonstration only."
    )
    with gr.Row():
        with gr.Column():
            fields = build_inputs()
            btn = gr.Button("Score applicant", variant="primary")
        with gr.Column():
            out_p = gr.Textbox(label="Probability of bad credit risk")
            out_d = gr.Textbox(label="Decision")
            out_t = gr.Textbox(label="Response time")
    btn.click(ui_predict, inputs=fields, outputs=[out_p, out_d, out_t])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
