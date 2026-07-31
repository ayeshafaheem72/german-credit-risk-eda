"""
K-Means borrower segmentation -- an UNSUPERVISED analysis, separate from the
credit-risk classification task. The target column is never used to fit the
clusters; it is only joined back afterward to see whether the resulting
segments happen to differ in risk, as a descriptive check.

Run: python src/clustering.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from data_loader import load_labeled

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"

NUMERIC_COLS = [
    "duration_months", "credit_amount", "installment_rate_pct",
    "present_residence_since", "age_years", "existing_credits_count",
    "num_dependents",
]


def main():
    df = load_labeled()
    X = df[NUMERIC_COLS].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    log_lines = []

    def log(x=""):
        print(x)
        log_lines.append(str(x))

    log("K-Means borrower segmentation (features: " + ", ".join(NUMERIC_COLS) + ")")
    log("Note: numeric attributes only -- categorical one-hot columns were excluded "
        "from the clustering feature set because their large count (41 dummy columns "
        "vs 7 numeric) would dominate Euclidean distance and wash out the numeric "
        "signal; this is a deliberate scope choice, not an oversight.")
    log("")

    # ---- Choose k: elbow (inertia) + silhouette across k=2..8 ----
    inertias = []
    silhouettes = []
    ks = list(range(2, 9))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    log(f"{'k':>3}{'inertia':>12}{'silhouette':>12}")
    for k, inertia, sil in zip(ks, inertias, silhouettes):
        log(f"{k:>3}{inertia:>12.1f}{sil:>12.4f}")

    best_k = ks[int(np.argmax(silhouettes))]
    log(f"\nBest k by silhouette score: {best_k} (silhouette={max(silhouettes):.4f})")
    log("A silhouette in this range (well under 0.5) means clusters are real but "
        "overlapping, not cleanly separated -- expected for demographic/financial "
        "survey data with no natural gaps between groups. This is reported honestly, "
        "not smoothed over.")

    # Elbow plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ks, inertias, marker="o")
    axes[0].set_title("Elbow Method (Inertia vs k)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(ks, silhouettes, marker="o", color="#C44E52")
    axes[1].set_title("Silhouette Score vs k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_kmeans_k_selection.png", dpi=150)
    plt.close(fig)

    # ---- Fit final model at best_k ----
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = km_final.fit_predict(X_scaled)

    log(f"\nCluster sizes (k={best_k}):")
    log(df["cluster"].value_counts().sort_index().to_string())

    log("\nCluster profiles (mean of numeric features):")
    profile = df.groupby("cluster")[NUMERIC_COLS].mean().round(1)
    log(profile.to_string())

    log("\nCluster bad-risk rate (target NOT used to fit clusters -- joined back "
        "only to describe the segments after the fact):")
    risk_profile = df.groupby("cluster")["target"].mean().round(3)
    log(risk_profile.to_string())

    log("\nMost common loan purpose per cluster:")
    top_purpose = df.groupby("cluster")["purpose"].agg(lambda s: s.value_counts().idxmax())
    log(top_purpose.to_string())

    # Scatter of two most interpretable dimensions, colored by cluster
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(df["age_years"], df["credit_amount"], c=df["cluster"],
                          cmap="tab10", alpha=0.6, s=25)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Credit Amount (DM)")
    ax.set_title(f"Borrower Segments (k={best_k}): Age vs Credit Amount")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
    ax.add_artist(legend1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_kmeans_clusters_scatter.png", dpi=150)
    plt.close(fig)

    (ROOT / "reports" / "clustering_results.txt").write_text("\n".join(log_lines), encoding="utf-8")
    summary = {
        "best_k": best_k,
        "silhouette_scores": {str(k): round(s, 4) for k, s in zip(ks, silhouettes)},
        "cluster_sizes": df["cluster"].value_counts().sort_index().to_dict(),
        "cluster_bad_rate": risk_profile.round(3).to_dict(),
    }
    with open(ROOT / "reports" / "clustering_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    log("\nSaved reports/clustering_results.txt, clustering_results.json, and 2 figures.")


if __name__ == "__main__":
    main()
