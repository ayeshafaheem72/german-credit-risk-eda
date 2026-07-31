"""
Baseline classification models for the German Credit risk task.

Models:
  - Logistic Regression (default, and class_weight='balanced')
  - Decision Tree (default, shows overfitting via train vs test gap)
  - Random Forest (default, and class_weight='balanced')

Metrics: accuracy, precision, recall, F1 (all for the "bad credit" / positive
class = 1, which is the class that actually matters to a lender), confusion
matrix, and a business-cost score using the cost matrix published with the
dataset (misclassifying an actual bad risk as good costs 5x more than the
reverse).

Run: python src/modeling.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

from preprocessing import get_train_test_split, build_preprocessor

ROOT = Path(__file__).resolve().parent.parent

# Cost matrix from the UCI documentation: rows=actual, cols=predicted.
# actual bad (1) predicted good (0) -> cost 5 (a missed defaulter)
# actual good (0) predicted bad (1) -> cost 1 (an unnecessarily declined good customer)
FN_COST = 5
FP_COST = 1


def business_cost(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(fn * FN_COST + fp * FP_COST)


def evaluate(name, y_true, y_pred, y_proba=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result = {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_bad": round(precision_score(y_true, y_pred, pos_label=1), 4),
        "recall_bad": round(recall_score(y_true, y_pred, pos_label=1), 4),
        "f1_bad": round(f1_score(y_true, y_pred, pos_label=1), 4),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
        "business_cost": business_cost(y_true, y_pred),
    }
    if y_proba is not None:
        result["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
    return result


def main():
    X_train, X_test, y_train, y_test = get_train_test_split()

    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    results = []
    log_lines = []

    def log(x=""):
        print(x)
        log_lines.append(str(x))

    # ---- Naive baseline: predict majority class every time ----
    majority_pred = np.zeros_like(y_test)
    results.append(evaluate("Majority-class baseline (always predict 'good')", y_test, majority_pred))

    # ---- Logistic Regression, default ----
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_enc, y_train)
    lr_pred = lr.predict(X_test_enc)
    lr_proba = lr.predict_proba(X_test_enc)[:, 1]
    results.append(evaluate("Logistic Regression (default)", y_test, lr_pred, lr_proba))

    # ---- Logistic Regression, class_weight balanced ----
    lr_bal = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_bal.fit(X_train_enc, y_train)
    lr_bal_pred = lr_bal.predict(X_test_enc)
    lr_bal_proba = lr_bal.predict_proba(X_test_enc)[:, 1]
    results.append(evaluate("Logistic Regression (class_weight='balanced')", y_test, lr_bal_pred, lr_bal_proba))

    # ---- Decision Tree, default (unrestricted depth -> expect overfit) ----
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_enc, y_train)
    dt_train_acc = accuracy_score(y_train, dt.predict(X_train_enc))
    dt_pred = dt.predict(X_test_enc)
    dt_res = evaluate("Decision Tree (default, unpruned)", y_test, dt_pred)
    dt_res["train_accuracy"] = round(dt_train_acc, 4)
    results.append(dt_res)

    # ---- Decision Tree, depth-limited ----
    dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")
    dt_pruned.fit(X_train_enc, y_train)
    dt_pruned_train_acc = accuracy_score(y_train, dt_pruned.predict(X_train_enc))
    dt_pruned_pred = dt_pruned.predict(X_test_enc)
    dt_pruned_proba = dt_pruned.predict_proba(X_test_enc)[:, 1]
    dt_pruned_res = evaluate("Decision Tree (max_depth=4, balanced)", y_test, dt_pruned_pred, dt_pruned_proba)
    dt_pruned_res["train_accuracy"] = round(dt_pruned_train_acc, 4)
    results.append(dt_pruned_res)

    # ---- Random Forest, default ----
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_enc, y_train)
    rf_pred = rf.predict(X_test_enc)
    rf_proba = rf.predict_proba(X_test_enc)[:, 1]
    results.append(evaluate("Random Forest (default, n_estimators=200)", y_test, rf_pred, rf_proba))

    # ---- Random Forest, balanced ----
    rf_bal = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf_bal.fit(X_train_enc, y_train)
    rf_bal_pred = rf_bal.predict(X_test_enc)
    rf_bal_proba = rf_bal.predict_proba(X_test_enc)[:, 1]
    results.append(evaluate("Random Forest (class_weight='balanced')", y_test, rf_bal_pred, rf_bal_proba))

    # ---- Report ----
    log("=" * 100)
    log(f"{'Model':<48}{'Acc':>7}{'Prec(bad)':>11}{'Rec(bad)':>10}{'F1(bad)':>9}{'ROC-AUC':>9}{'Cost':>7}")
    log("=" * 100)
    for r in results:
        auc = r.get("roc_auc", float("nan"))
        log(f"{r['model']:<48}{r['accuracy']:>7.3f}{r['precision_bad']:>11.3f}"
            f"{r['recall_bad']:>10.3f}{r['f1_bad']:>9.3f}{auc:>9.3f}{r['business_cost']:>7}")

    log("\nConfusion matrices ([[TN, FP], [FN, TP]], positive class = 1 = bad risk):")
    for r in results:
        log(f"  {r['model']}: {r['confusion_matrix']}")

    log("\nDecision Tree overfitting check:")
    log(f"  Unpruned tree -- train accuracy: {results[3]['train_accuracy']:.4f}, "
        f"test accuracy: {results[3]['accuracy']:.4f} "
        f"(gap = {results[3]['train_accuracy'] - results[3]['accuracy']:.4f})")
    log(f"  Depth=4 tree  -- train accuracy: {results[4]['train_accuracy']:.4f}, "
        f"test accuracy: {results[4]['accuracy']:.4f} "
        f"(gap = {results[4]['train_accuracy'] - results[4]['accuracy']:.4f})")

    log("\nHonest read:")
    log("  - The majority-class baseline already scores 70% accuracy by predicting "
        "'good' for everyone -- any model must be judged against that floor, not against 0%.")
    log("  - The unpruned Decision Tree overfits clearly (train accuracy near-perfect, "
        "large train/test gap); it is not a usable model as-is.")
    log("  - class_weight='balanced' consistently trades accuracy for recall on the "
        "bad-risk class -- exactly the trade a lender using the published cost matrix "
        "(missed defaulter costs 5x a wrongly-declined good customer) should want, but it "
        "is a real trade, not a free lunch: precision on 'bad' drops when recall rises.")
    log("  - Business cost (5*FN + 1*FP on 200 test cases) is reported alongside standard "
        "metrics because accuracy/F1 alone do not reflect this dataset's stated cost "
        "asymmetry -- a model can have a *worse* F1 but a *lower* business cost.")

    (ROOT / "reports" / "model_results.txt").write_text("\n".join(log_lines), encoding="utf-8")
    with open(ROOT / "reports" / "model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log("\nSaved reports/model_results.txt and reports/model_results.json")

    # detailed classification report for the two most relevant models
    log("\nFull classification report -- Logistic Regression (balanced):")
    log(classification_report(y_test, lr_bal_pred, target_names=["good(0)", "bad(1)"]))
    log("Full classification report -- Random Forest (default):")
    log(classification_report(y_test, rf_pred, target_names=["good(0)", "bad(1)"]))


if __name__ == "__main__":
    main()
