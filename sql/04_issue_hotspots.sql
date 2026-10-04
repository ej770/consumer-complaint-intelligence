-- Issue hotspots: high-volume issues ranked by how often they end in relief
SELECT
    product,
    issue,
    COUNT(*)                                AS complaints,
    ROUND(100.0 * AVG(relief), 1)           AS relief_rate_pct,
    ROUND(100.0 * AVG(monetary_relief), 1)  AS monetary_relief_pct
FROM complaints
GROUP BY product, issue
HAVING COUNT(*) >= 1000
ORDER BY relief_rate_pct DESC;
