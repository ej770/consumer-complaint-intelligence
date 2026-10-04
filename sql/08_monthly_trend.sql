-- Monthly volume and relief rate (complete months only; narratives publish with a lag)
SELECT
    month,
    COUNT(*)                                AS complaints,
    SUM(CASE WHEN product = 'Credit reporting' THEN 1 ELSE 0 END) AS credit_reporting,
    ROUND(100.0 * AVG(relief), 1)           AS relief_rate_pct
FROM complaints
WHERE complete_month = 1
GROUP BY month
ORDER BY month;
