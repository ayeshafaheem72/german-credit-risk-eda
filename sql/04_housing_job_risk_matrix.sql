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
