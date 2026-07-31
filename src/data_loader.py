"""
Load the UCI Statlog (German Credit Data) dataset and map its coded
categorical levels to human-readable labels.

Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
Original contributor: Prof. Dr. Hans Hofmann, Universitat Hamburg.
File used: german.data (the categorical/symbolic version, 1000 rows x 20
attributes + target). Downloaded 2026-07-31.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_FILE = DATA_DIR / "german.data"

COLUMN_NAMES = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors_guarantors",
    "present_residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "target",  # 1 = good credit risk, 2 = bad credit risk
]

# Human-readable mappings transcribed directly from german.doc (UCI).
CODE_MAPS = {
    "checking_account_status": {
        "A11": "< 0 DM",
        "A12": "0-200 DM",
        "A13": ">= 200 DM",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken / all paid duly",
        "A31": "all credits at this bank paid duly",
        "A32": "existing credits paid duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits elsewhere",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "other",
    },
    "savings_account": {
        "A61": "< 100 DM",
        "A62": "100-500 DM",
        "A63": "500-1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown / no savings account",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1-4 years",
        "A74": "4-7 years",
        "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male: divorced/separated",
        "A92": "female: divorced/separated/married",
        "A93": "male: single",
        "A94": "male: married/widowed",
        "A95": "female: single",
    },
    "other_debtors_guarantors": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property": {
        "A121": "real estate",
        "A122": "building society savings / life insurance",
        "A123": "car or other",
        "A124": "unknown / no property",
    },
    "other_installment_plans": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "for free",
    },
    "job": {
        "A171": "unemployed/unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled employee/official",
        "A174": "management/self-employed/highly qualified",
    },
    "telephone": {
        "A191": "none",
        "A192": "yes, registered",
    },
    "foreign_worker": {
        "A201": "yes",
        "A202": "no",
    },
}


def load_raw() -> pd.DataFrame:
    """Read the whitespace-delimited german.data file with real column names."""
    df = pd.read_csv(RAW_FILE, sep=r"\s+", header=None, names=COLUMN_NAMES, engine="python")
    return df


def load_labeled(binary_target: bool = True) -> pd.DataFrame:
    """
    Load the data and decode categorical codes (A11, A34, ...) into readable
    labels. Recodes target from {1=good, 2=bad} to {0=good, 1=bad} so that
    "1" consistently means the positive/at-risk class for sklearn metrics.
    """
    df = load_raw()
    for col, mapping in CODE_MAPS.items():
        df[col] = df[col].map(mapping)
    if binary_target:
        df["target"] = df["target"].map({1: 0, 2: 1})  # 0 = good, 1 = bad
    return df


if __name__ == "__main__":
    data = load_labeled()
    print(data.shape)
    print(data.head())
    print(data["target"].value_counts())
