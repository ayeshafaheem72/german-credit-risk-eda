"""Measure single-request inference latency over the whole dataset (p50, p95, max)."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / "model.joblib")
df = pd.read_csv(ROOT.parent / "data" / "german_credit_clean.csv").drop(columns="target")

times = []
for i in range(len(df)):
    row = df.iloc[[i]]
    t = time.perf_counter()
    model.predict_proba(row)
    times.append((time.perf_counter() - t) * 1000)

result = {"requests": len(times), "p50_ms": round(float(np.percentile(times, 50)), 2),
          "p95_ms": round(float(np.percentile(times, 95)), 2), "max_ms": round(max(times), 2)}
(ROOT / "latency_benchmark.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
