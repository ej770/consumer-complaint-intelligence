-- Texas spotlight: how Texas complaints compare with the rest of the U.S., by product
SELECT
    product,
    SUM(is_texas)                                                        AS texas_complaints,
    SUM(is_houston)                                                      AS houston_complaints,
    ROUND(100.0 * SUM(is_texas) / SUM(SUM(is_texas)) OVER (), 1)          AS texas_mix_pct,
    ROUND(100.0 * SUM(1 - is_texas) / SUM(SUM(1 - is_texas)) OVER (), 1)  AS rest_of_us_mix_pct,
    ROUND(100.0 * AVG(CASE WHEN is_texas = 1 THEN relief END), 1)         AS texas_relief_pct,
    ROUND(100.0 * AVG(CASE WHEN is_texas = 0 THEN relief END), 1)         AS rest_of_us_relief_pct
FROM complaints
GROUP BY product
ORDER BY texas_complaints DESC;
