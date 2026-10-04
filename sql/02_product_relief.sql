-- Volume and outcome by product: where do consumers actually get relief?
SELECT
    product,
    COUNT(*)                                                    AS complaints,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)          AS share_pct,
    ROUND(100.0 * AVG(relief), 1)                               AS relief_rate_pct,
    ROUND(100.0 * AVG(monetary_relief), 1)                      AS monetary_relief_pct,
    ROUND(100.0 * AVG(timely), 1)                               AS timely_pct,
    CAST(ROUND(AVG(narrative_words)) AS INTEGER)                AS avg_words
FROM complaints
GROUP BY product
ORDER BY complaints DESC;
