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
