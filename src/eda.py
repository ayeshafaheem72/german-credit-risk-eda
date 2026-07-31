"""
Exploratory Data Analysis for the UCI German Credit dataset.
Run: python src/eda.py
Produces printed summary stats (captured into reports/eda_summary.txt)
and 4 figures in reports/figures/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from data_loader import load_labeled

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")

NUMERIC_COLS = [
    "duration_months", "credit_amount", "installment_rate_pct",
    "present_residence_since", "age_years", "existing_credits_count",
    "num_dependents",
]


def main():
    df = load_labeled()
    lines = []

    def log(x=""):
        print(x)
        lines.append(str(x))

    log("=" * 70)
    log("1. STRUCTURE")
    log("=" * 70)
    log(f"Shape: {df.shape}")
    log("\nDtypes:")
    log(df.dtypes.to_string())

    log("\n" + "=" * 70)
    log("2. MISSING VALUES")
    log("=" * 70)
    missing = df.isnull().sum()
    log(missing.to_string())
    log(f"\nTotal missing cells: {missing.sum()} out of {df.size} ({missing.sum()/df.size:.2%})")
    log("Note: the UCI German Credit dataset ships with no missing values by "
        "construction (Hofmann pre-cleaned it for StatLog). This is verified "
        "above, not assumed -- a real-world pull from a live banking system "
        "would not be this clean.")

    log("\n" + "=" * 70)
    log("3. CLASS BALANCE (target: 0=good credit risk, 1=bad credit risk)")
    log("=" * 70)
    counts = df["target"].value_counts()
    pct = df["target"].value_counts(normalize=True) * 100
    log(counts.to_string())
    log(pct.round(2).to_string())
    log("This is a 70/30 imbalance, not extreme, but enough that accuracy alone "
        "is a misleading metric -- a model predicting 'good' for everyone would "
        "score 70% accuracy while catching zero defaulters.")

    log("\n" + "=" * 70)
    log("4. NUMERIC DISTRIBUTIONS (describe)")
    log("=" * 70)
    log(df[NUMERIC_COLS].describe().T.to_string())

    log("\n" + "=" * 70)
    log("5. NUMERIC FEATURE CORRELATION WITH TARGET (point-biserial via .corr)")
    log("=" * 70)
    corr_with_target = df[NUMERIC_COLS + ["target"]].corr()["target"].drop("target").sort_values(key=abs, ascending=False)
    log(corr_with_target.round(3).to_string())
    log("All correlations are weak in isolation (|r| < 0.35) -- expected, since "
        "credit risk here is driven by categorical factors (checking account "
        "status, credit history, purpose) more than by single numeric fields.")

    log("\n" + "=" * 70)
    log("6. CATEGORICAL SNAPSHOT: default rate by checking_account_status")
    log("=" * 70)
    rate_tbl = df.groupby("checking_account_status")["target"].agg(["mean", "count"])
    rate_tbl = rate_tbl.rename(columns={"mean": "bad_rate"}).sort_values("bad_rate", ascending=False)
    log(rate_tbl.round(3).to_string())

    # ---- Figures ----

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.sort_index().plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_xticklabels(["Good (0)", "Bad (1)"], rotation=0)
    ax.set_title("Class Balance: Credit Risk Target")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.sort_index()):
        ax.text(i, v + 5, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_balance.png", dpi=150)
    plt.close(fig)

    # 2. Credit amount distribution by target
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x="credit_amount", hue="target", bins=40, kde=True,
                  palette={0: "#4C72B0", 1: "#C44E52"}, ax=ax)
    ax.set_title("Credit Amount Distribution by Risk Class")
    ax.set_xlabel("Credit Amount (DM)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_credit_amount_by_target.png", dpi=150)
    plt.close(fig)

    # 3. Age distribution by target (boxplot)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=df, x="target", y="age_years", hue="target",
                palette={0: "#4C72B0", 1: "#C44E52"}, legend=False, ax=ax)
    ax.set_xticklabels(["Good (0)", "Bad (1)"])
    ax.set_title("Age Distribution by Risk Class")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_age_by_target.png", dpi=150)
    plt.close(fig)

    # 4. Correlation heatmap of numeric features
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[NUMERIC_COLS + ["target"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix: Numeric Features + Target")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 5. Bad rate by checking account status (categorical predictor)
    fig, ax = plt.subplots(figsize=(7, 4))
    rate_tbl["bad_rate"].sort_values().plot(kind="barh", ax=ax, color="#C44E52")
    ax.set_title("Default (Bad Risk) Rate by Checking Account Status")
    ax.set_xlabel("Bad Rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_bad_rate_by_checking_status.png", dpi=150)
    plt.close(fig)

    (ROOT / "reports" / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nSaved 5 figures to {FIG_DIR}")
    log("Saved text summary to reports/eda_summary.txt")


if __name__ == "__main__":
    main()
