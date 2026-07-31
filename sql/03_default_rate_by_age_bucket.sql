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
