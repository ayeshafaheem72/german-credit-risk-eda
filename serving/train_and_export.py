"""Train the chosen credit-risk model and export it for serving.

Chosen model: Logistic Regression with class_weight='balanced', the lowest business cost
in reports/model_results.txt (missed defaulter costs 5x a wrongly declined good customer).
The preprocessing is bundled into one sklearn Pipeline so serving cannot drift from training.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import NUMERIC_COLS, CATEGORICAL_COLS, build_preprocessor  # noqa: E402

MODEL_PATH = ROOT / "serving" / "model.joblib"
META_PATH = ROOT / "serving" / "model_meta.json"


def load_data():
    df = pd.read_csv(ROOT / "data" / "german_credit_clean.csv")
    X, y = df[NUMERIC_COLS + CATEGORICAL_COLS], df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main():
    X_train, X_test, y_train, y_test = load_data()
    model = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    meta = {
        "model": "LogisticRegression(class_weight='balanced') + one-hot/scaling pipeline",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_rows": int(len(y_test)),
        "recall_bad": round(recall_score(y_test, pred), 4),
        "precision_bad": round(precision_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "business_cost": int(5 * fn + fp),
    }
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
