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
