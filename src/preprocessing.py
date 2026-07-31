"""
Preprocessing for the German Credit dataset.

Decisions made here (see reports/preprocessing_notes.md for the write-up):
1. Missing values: none exist in this dataset (confirmed in EDA) -- no
   imputation logic is needed, and none is added speculatively.
2. Categorical encoding: one-hot encoding (drop='first' to avoid the dummy
   trap) for all 13 categorical attributes. Several of them (checking
   account status, savings account, employment length) are ordinal in
   principle, but German Credit's ordinal bins are not evenly spaced in
   real terms (e.g. "0-200 DM" vs ">= 200 DM" is not a fixed step), so
   treating them as nominal is the safer default. Noted as a possible
   future refinement, not treated as free.
3. Scaling: StandardScaler on the 7 numeric attributes. Required for
   Logistic Regression (distance/gradient-based); harmless but unnecessary
   for the tree-based models (splits are scale-invariant). Applying the
   same transformed feature set to both models keeps the comparison clean
   and avoids maintaining two parallel feature tables.
4. Split: stratified 80/20 train/test split on the target, random_state=42,
   so the 70/30 class ratio is preserved in both folds.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from data_loader import load_labeled

NUMERIC_COLS = [
    "duration_months", "credit_amount", "installment_rate_pct",
    "present_residence_since", "age_years", "existing_credits_count",
    "num_dependents",
]
CATEGORICAL_COLS = [
    "checking_account_status", "credit_history", "purpose", "savings_account",
    "employment_since", "personal_status_sex", "other_debtors_guarantors",
    "property", "other_installment_plans", "housing", "job", "telephone",
    "foreign_worker",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


def get_train_test_split(random_state: int = 42, test_size: float = 0.2):
    df = load_labeled()
    assert df.isnull().sum().sum() == 0, "unexpected missing values found"

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    return list(preprocessor.get_feature_names_out())


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_split()
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train class balance:\n", y_train.value_counts(normalize=True).round(3))
    print("Test class balance:\n", y_test.value_counts(normalize=True).round(3))

    pre = build_preprocessor()
    pre.fit(X_train)
    print("Encoded feature count:", len(get_feature_names(pre)))
