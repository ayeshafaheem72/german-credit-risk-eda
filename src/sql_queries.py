"""
Loads the cleaned German Credit data into a local SQLite database and runs
SQL queries against it to answer exploratory questions -- demonstrating SQL
proficiency on top of the pandas-based EDA.

Run: python src/sql_queries.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

from data_loader import load_labeled

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "german_credit.db"
SQL_DIR = ROOT / "sql"

QUERIES = {
    "01_default_rate_by_checking_status": """
        -- Q1: Which checking-account status carries the highest default rate,
        -- and is it backed by enough volume to trust?
        SELECT
            checking_account_status,
            COUNT(*)                                   AS n_customers,
            SUM(target)                                AS n_bad,
            ROUND(AVG(target * 1.0), 3)                AS bad_rate
        FROM credit
        GROUP BY checking_account_status
        ORDER BY bad_rate DESC;
    """,
    "02_avg_amount_duration_by_purpose": """
        -- Q2: For each loan purpose, what's the average credit amount,
        -- average duration, and default rate? Only purposes with at least
        -- 20 customers are shown, to avoid drawing conclusions from tiny groups.
        SELECT
            purpose,
            COUNT(*)                                   AS n_customers,
            ROUND(AVG(credit_amount), 0)                AS avg_credit_amount,
            ROUND(AVG(duration_months), 1)               AS avg_duration_months,
            ROUND(AVG(target * 1.0), 3)                 AS bad_rate
        FROM credit
        GROUP BY purpose
        HAVING COUNT(*) >= 20
        ORDER BY bad_rate DESC;
    """,
    "03_default_rate_by_age_bucket": """
        -- Q3: Does default risk vary by age bracket?
        SELECT
            CASE
                WHEN age_years < 25 THEN 'under 25'
                WHEN age_years < 35 THEN '25-34'
                WHEN age_years < 45 THEN '35-44'
                WHEN age_years < 55 THEN '45-54'
                ELSE '55+'
            END                                          AS age_bucket,
            COUNT(*)                                    AS n_customers,
            ROUND(AVG(target * 1.0), 3)                 AS bad_rate,
            ROUND(AVG(credit_amount), 0)                AS avg_credit_amount
        FROM credit
        GROUP BY age_bucket
        ORDER BY age_bucket;
    """,
    "04_housing_job_risk_matrix": """
        -- Q4 (bonus): Cross-tabulate housing x job to find the riskiest
        -- combination that still has meaningful sample size (>=15).
        SELECT
            housing,
            job,
            COUNT(*)                                    AS n_customers,
            ROUND(AVG(target * 1.0), 3)                 AS bad_rate
        FROM credit
        GROUP BY housing, job
        HAVING COUNT(*) >= 15
        ORDER BY bad_rate DESC
        LIMIT 10;
    """,
}


def main():
    df = load_labeled()

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("credit", conn, if_exists="replace", index=False)

    log_lines = []

    def log(x=""):
        print(x)
        log_lines.append(str(x))

    SQL_DIR.mkdir(exist_ok=True)

    for name, sql in QUERIES.items():
        (SQL_DIR / f"{name}.sql").write_text(sql.strip() + "\n", encoding="utf-8")

        log("=" * 90)
        log(name)
        log("=" * 90)
        result = pd.read_sql_query(sql, conn)
        log(result.to_string(index=False))
        log("")

    conn.close()

    (ROOT / "reports" / "sql_query_results.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"SQLite DB written to {DB_PATH}")
    log(f"{len(QUERIES)} .sql files written to {SQL_DIR}")
    log("Results saved to reports/sql_query_results.txt")


if __name__ == "__main__":
    main()
