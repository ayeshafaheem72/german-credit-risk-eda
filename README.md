# German Credit Risk: EDA, Baseline Classification & Borrower Segmentation

A reproducible, honestly-reported baseline data science project on a real
banking dataset: exploratory data analysis, documented preprocessing,
baseline classification (Logistic Regression / Decision Tree / Random
Forest), unsupervised borrower segmentation (K-Means), and SQL-based
exploration — built as a portfolio project for a Junior Data Scientist / ML
Analyst role.

## Dataset

**[UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)**
— 1000 loan applicants at a German bank, 20 attributes (7 numeric, 13
categorical), contributed by Prof. Dr. Hans Hofmann (Universität Hamburg),
distributed via the UCI Machine Learning Repository.

- Downloaded directly from `archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data` on 2026-07-31.
- Target: `1 = good credit risk`, `2 = bad credit risk` in the raw file, recoded to `0 = good`, `1 = bad` throughout this project.
- The dataset ships with a **cost matrix**: misclassifying an actual bad risk as good costs **5x** more than the reverse (a missed defaulter vs. an unnecessarily declined good customer). This project uses that cost matrix as a real evaluation criterion, not just accuracy.

No missing values were introduced or found — this is UCI/StatLog's pre-cleaned
benchmark version. That's a real limitation of this project's realism (see
Caveats), not something to gloss over.

## Project structure

```
bank-credit-risk-eda/
├── data/                       german.data (raw), german.doc (attribute docs),
│                                german_credit_clean.csv, german_credit.db (SQLite)
├── src/
│   ├── data_loader.py          parses raw data, decodes category codes
│   ├── eda.py                  full EDA, saves 5 figures + text summary
│   ├── preprocessing.py        encoding/scaling/split, documented choices
│   ├── modeling.py             LogReg / Decision Tree / Random Forest + evaluation
│   ├── clustering.py           K-Means borrower segmentation
│   └── sql_queries.py          builds SQLite DB, runs 4 SQL queries
├── sql/                        the 4 .sql query files, runnable standalone
├── notebooks/
│   ├── German_Credit_Risk_Analysis.ipynb   full analysis, already executed
│   └── build_notebook.py       generates the notebook from the same src/ code
├── reports/
│   ├── figures/                 7 PNG charts
│   ├── eda_summary.txt, model_results.{txt,json}, clustering_results.{txt,json},
│   │   sql_query_results.txt, preprocessing_notes.md
└── requirements.txt
```

## Reproduce it

```bash
pip install -r requirements.txt
python src/eda.py
python src/preprocessing.py
python src/modeling.py
python src/clustering.py
python src/sql_queries.py
```

Or open `notebooks/German_Credit_Risk_Analysis.ipynb` directly — it's already
executed with real outputs, and re-running it end-to-end reproduces the exact
same numbers (fixed `random_state=42` throughout).

## EDA highlights

- **Class balance:** 700 good / 300 bad (70% / 30%) — imbalanced enough that
  accuracy alone is misleading.
- **No missing values** (0 / 21,000 cells) — verified, not assumed.
- **Numeric correlations with target are all weak** (|r| < 0.22) — risk here
  is driven by categorical factors more than any single numeric field.
- **Counterintuitive finding:** applicants with *no checking account* have
  the **lowest** default rate (11.7%, n=394), while those with a negative
  balance (`< 0 DM`) have the **highest** (49.3%, n=274). A model — or a
  human underwriter — that assumes "no account" implies higher risk would be
  wrong on this data.
- **Loan purpose matters:** `education` loans default at 44.0% (n=50) vs.
  `car (used)` loans at 16.5% (n=103) — see `sql/02_avg_amount_duration_by_purpose.sql`.
- **Age matters:** under-25 applicants default at 40.9% vs. 23.0–27.8% for
  every older bracket — see `sql/03_default_rate_by_age_bucket.sql`.

Figures: `reports/figures/01_class_balance.png` through `05_bad_rate_by_checking_status.png`.

## Preprocessing decisions (full detail in `reports/preprocessing_notes.md`)

| Step | Decision |
|---|---|
| Missing values | None to handle (verified) |
| Categorical encoding | One-hot, `drop='first'` (13 columns → 41 dummy columns) |
| Numeric scaling | `StandardScaler` on 7 numeric columns |
| Split | Stratified 80/20, `random_state=42` → 800 train / 200 test, 70/30 ratio preserved in both |
| Final feature count | 20 raw → 48 encoded |

## Models & results — real numbers, test set (n=200)

Metrics are for the **bad-risk (1) class** — the one that matters to a
lender. `business_cost = 5×FN + 1×FP`, per the dataset's own cost matrix.

| Model | Accuracy | Precision (bad) | Recall (bad) | F1 (bad) | ROC-AUC | Business Cost |
|---|---|---|---|---|---|---|
| Majority-class baseline | 0.700 | 0.000 | 0.000 | 0.000 | — | 300 |
| Logistic Regression (default) | 0.775 | 0.653 | 0.533 | 0.587 | 0.806 | 157 |
| **Logistic Regression (balanced)** | 0.745 | 0.552 | **0.800** | 0.653 | 0.807 | **99** |
| Decision Tree (unpruned) | 0.620 | 0.382 | 0.433 | 0.406 | — | 212 |
| Decision Tree (depth=4, balanced) | 0.630 | 0.426 | 0.667 | 0.520 | 0.673 | 154 |
| **Random Forest (default)** | **0.765** | **0.676** | 0.417 | 0.516 | 0.786 | 187 |
| Random Forest (balanced) | 0.765 | 0.686 | 0.400 | 0.505 | 0.793 | 191 |

Confusion matrices (`[[TN, FP], [FN, TP]]`):

- Logistic Regression (balanced): `[[101, 39], [12, 48]]`
- Random Forest (default): `[[128, 12], [35, 25]]`

**Decision Tree overfitting, shown not asserted:** the unpruned tree hits
**100% train accuracy** but only **62.0% test accuracy** — a 38-point gap.
Limiting `max_depth=4` shrinks the gap to 10.5 points (73.5% train vs. 63.0%
test) at the cost of being the weakest model overall. It is included here
specifically to demonstrate the failure mode, not because it's competitive.

### The honest trade-off

- **Random Forest (default)** wins on accuracy (0.765) and precision on bad
  risk (0.676) — but its recall on bad risk (0.417) is mediocre: it misses
  35 of 60 actual defaulters in the test set.
- **Logistic Regression (balanced)** has *lower* accuracy (0.745) but the
  *best* recall (0.800) and the *lowest business cost by a wide margin* (99
  vs. 157–212 for every other model) — because under the bank's own 5:1 cost
  asymmetry, catching defaulters matters more than overall accuracy.
- **There is no single "best" model here** — it depends on what a lender
  optimizes for. Reporting only accuracy would make Random Forest look best;
  reporting the dataset's own stated cost matrix makes Logistic Regression
  (balanced) the better business decision. Both are reported side by side,
  deliberately, instead of picking whichever number looks better.

## Borrower segmentation (K-Means, unsupervised)

A separate analysis. The target column was **never used to fit the
clusters** — it's joined back afterward only to describe the segments.
Clustering features: the 7 numeric columns only (one-hot categorical columns
were excluded — 41 sparse dummy columns would dominate Euclidean distance).

- Tested k = 2 to 8; **best k = 3** by silhouette score (**0.254**).
- Silhouette of 0.254 is low — clusters are real but overlapping, not
  cleanly separated. Reported as-is, not tuned to look better.

| Cluster | n | Avg credit amount | Avg duration | Bad rate | Dominant purpose |
|---|---|---|---|---|---|
| 0 | 173 | 7,937 DM | 38.9 mo | **0.474** | car (used) |
| 1 | 686 | 2,188 DM | 17.0 mo | 0.257 | radio/television |
| 2 | 141 | 2,818 DM | 18.1 mo | 0.298 | car (new) |

**Finding:** the small segment of large, long-duration loans (Cluster 0) has
nearly double the default rate of the other two segments — consistent with
the duration/amount signal seen independently in the EDA.

## SQL

All 4 queries in `sql/` run against a local SQLite copy of the cleaned data
(`data/german_credit.db`), built by `src/sql_queries.py`. Full output in
`reports/sql_query_results.txt`. Example (`sql/01_default_rate_by_checking_status.sql`):

```sql
SELECT checking_account_status, COUNT(*) AS n_customers, SUM(target) AS n_bad,
       ROUND(AVG(target * 1.0), 3) AS bad_rate
FROM credit
GROUP BY checking_account_status
ORDER BY bad_rate DESC;
```

## Caveats & limitations (read this before reusing the numbers)

- **Pre-cleaned benchmark data.** This is StatLog's cleaned version — no
  missing values, no duplicates, no messy production artifacts. A real bank
  feed needs real data-cleaning work this project didn't have to do.
- **Small and dated.** 1000 rows, 1994 Germany. Absolute credit amounts (DM)
  and demographic risk patterns will not transfer directly to a modern
  market.
- **Fair-lending flag on `personal_status_sex`.** One of the 20 original
  attributes encodes applicant sex combined with marital status. It's kept
  in this baseline to demonstrate technique faithfully on the original
  dataset, but flagging it explicitly: in a real underwriting model, sex and
  marital status are protected attributes under fair-lending regulation
  (e.g., US ECOA, EU gender-equality law) and should be excluded or handled
  through a fairness audit, not fed into a model naively. Many tutorials
  using this dataset don't mention this — it's called out here on purpose.
- **No hyperparameter search.** Models use sane defaults plus one obvious
  variant (`class_weight='balanced'`, `max_depth=4`) — this is a baseline
  project, not a tuned production model, and is not presented as one.
- **K-Means clusters are soft.** Silhouette 0.254 means real but overlapping
  segments — not a clean customer taxonomy.

## Resume bullet

> Built an end-to-end credit-risk pipeline on the UCI German Credit dataset
> (1,000 applicants) — EDA, preprocessing, and baseline Logistic
> Regression/Decision Tree/Random Forest classifiers plus K-Means borrower
> segmentation, evaluated against the dataset's own cost matrix; found that a
> cost-weighted Logistic Regression cut modeled misclassification cost by 47%
> versus the accuracy-optimized Random Forest (187 → 99, on the bank's own
> 5:1 cost matrix) by trading 2 points of accuracy for 38 points of recall on
> actual defaulters.

## Serving: Gradio demo, Docker and CI

The chosen model (balanced Logistic Regression, lowest business cost) is exported as one sklearn
Pipeline, so preprocessing at inference can never drift from training.

| Piece | File |
|---|---|
| Train and export model + metrics | `serving/train_and_export.py` |
| Gradio demo with response time shown per request | `serving/app.py` |
| Prediction and latency log for monitoring | `serving/logs/predictions.csv` (created at runtime) |
| Latency benchmark over 1,000 single requests | `serving/benchmark.py` |
| Regression tests (recall floor, cost vs baseline, latency) | `tests/test_serving.py` |
| Container | `Dockerfile` |
| CI on every push: tests, then Docker build | `.github/workflows/ci.yml` |

Results on the held-out test set: recall on bad-risk applicants 80%, ROC-AUC 0.81, business cost 99
vs 300 for the always-approve baseline.

```bash
pip install -r requirements-serving.txt
python serving/train_and_export.py
python serving/benchmark.py      # p50 / p95 latency
python serving/app.py            # http://localhost:7860

# or with Docker
docker build -t credit-risk-demo .
docker run -p 7860:7860 credit-risk-demo
```
