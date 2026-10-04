-- Company benchmark: the 15 most-complained-about companies vs. everyone else
WITH by_company AS (
    SELECT
        company_group                               AS company,
        COUNT(*)                                    AS complaints,
        AVG(relief)                                 AS relief_rate,
        AVG(monetary_relief)                        AS monetary_rate,
        AVG(timely)                                 AS timely_rate
    FROM complaints
    GROUP BY company_group
),
main_product AS (
    SELECT company_group, product,
           ROW_NUMBER() OVER (PARTITION BY company_group ORDER BY COUNT(*) DESC) AS rn
    FROM complaints
    GROUP BY company_group, product
)
SELECT
    b.company,
    b.complaints,
    ROUND(100.0 * b.complaints / (SELECT COUNT(*) FROM complaints), 1) AS share_pct,
    ROUND(100.0 * b.relief_rate, 1)                 AS relief_rate_pct,
    ROUND(100.0 * b.monetary_rate, 1)               AS monetary_relief_pct,
    ROUND(100.0 * b.timely_rate, 1)                 AS timely_pct,
    m.product                                       AS main_product
FROM by_company b
JOIN main_product m ON m.company_group = b.company AND m.rn = 1
ORDER BY (b.company = 'All other companies'), b.complaints DESC;
