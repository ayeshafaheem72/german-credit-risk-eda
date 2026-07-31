# Preprocessing Decisions

| Step | Decision | Rationale |
|---|---|---|
| Missing values | None handled | Verified in EDA: 0 missing cells across all 21 columns / 1000 rows. No imputation logic added on spec. |
| Categorical encoding | One-hot encoding, `drop='first'` | 13 categorical attributes. Some (checking account status, savings account, employment length) are ordinal in principle, but the bin widths are uneven in real terms (e.g. "0-200 DM" vs ">=200 DM"), so nominal one-hot encoding is the safer default rather than assuming a linear ordinal scale. `drop='first'` avoids the dummy-variable trap for the linear model. |
| Numeric scaling | `StandardScaler` on 7 numeric columns | Required for Logistic Regression (a distance/gradient-based model). Has no effect on Decision Tree / Random Forest splits (scale-invariant), so using the same transformed matrix for both models keeps the comparison apples-to-apples without maintaining two parallel feature tables. |
| Train/test split | Stratified 80/20, `random_state=42` | Preserves the 70/30 good/bad class ratio in both folds (confirmed: train 700/300 -> 560/240, test 700/300 -> 140/60). Stratification matters here because the class imbalance is non-trivial. |
| Feature count after encoding | 20 raw -> 48 encoded columns | 7 numeric (scaled) + 41 one-hot dummy columns from 13 categorical attributes. |

## Known limitation
One-hot encoding the ordinal attributes discards their order. A follow-up
iteration could test ordinal/label encoding for `checking_account_status`,
`savings_account`, and `employment_since` and compare model performance --
noted here rather than silently assumed to be a non-issue.
