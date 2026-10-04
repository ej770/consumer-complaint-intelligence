-- Headline KPIs for the whole extract
SELECT
    COUNT(*)                                        AS complaints,
    COUNT(DISTINCT company)                         AS companies,
    ROUND(100.0 * AVG(relief), 1)                   AS relief_rate_pct,
    ROUND(100.0 * AVG(monetary_relief), 1)          AS monetary_relief_pct,
    ROUND(100.0 * AVG(timely), 1)                   AS timely_response_pct,
    ROUND(100.0 * AVG(is_duplicate_text), 1)        AS duplicate_text_pct,
    SUM(is_texas)                                   AS texas_complaints,
    SUM(is_houston)                                 AS houston_complaints,
    MIN(date_received)                              AS first_received,
    MAX(date_received)                              AS last_received
FROM complaints;
