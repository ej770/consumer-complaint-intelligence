-- Does the way consumers write change the outcome?
-- Raw relief rates can mislead because some language clusters in low- or high-relief products
-- (e.g. lawyers are mentioned most in mortgage and debt collection complaints).
-- "expected_pct" applies the product-level relief rates of complaints WITHOUT the signal to the
-- product mix of complaints WITH it (direct standardization), so the gap below is mix-adjusted.
WITH signals AS (
    SELECT complaint_id, product, relief, 'Mentions an attorney or lawyer' AS signal, mentions_attorney AS flag FROM complaints
    UNION ALL SELECT complaint_id, product, relief, 'Mentions suing or court',          mentions_legal_action   FROM complaints
    UNION ALL SELECT complaint_id, product, relief, 'Mentions a regulator (CFPB, AG, FTC, BBB)', mentions_regulator FROM complaints
    UNION ALL SELECT complaint_id, product, relief, 'Cites a consumer law (FCRA, FDCPA)', cites_law            FROM complaints
    UNION ALL SELECT complaint_id, product, relief, 'Mentions identity theft',          mentions_identity_theft FROM complaints
    UNION ALL SELECT complaint_id, product, relief, 'Mentions fees or overdraft',       mentions_fees           FROM complaints
),
baseline AS (   -- product-level relief rate among complaints without the signal
    SELECT signal, product, AVG(relief) AS rate_without
    FROM signals WHERE flag = 0
    GROUP BY signal, product
)
SELECT
    s.signal,
    SUM(s.flag)                                                         AS complaints_with_signal,
    ROUND(100.0 * AVG(s.flag), 1)                                       AS share_pct,
    ROUND(100.0 * AVG(CASE WHEN s.flag = 1 THEN s.relief END), 1)       AS relief_with_pct,
    ROUND(100.0 * AVG(CASE WHEN s.flag = 0 THEN s.relief END), 1)       AS relief_without_pct,
    ROUND(100.0 * AVG(CASE WHEN s.flag = 1 THEN b.rate_without END), 1) AS expected_pct,
    ROUND(100.0 * (AVG(CASE WHEN s.flag = 1 THEN s.relief END)
                 - AVG(CASE WHEN s.flag = 1 THEN b.rate_without END)), 1) AS mix_adjusted_gap_pts
FROM signals s
JOIN baseline b ON b.signal = s.signal AND b.product = s.product
GROUP BY s.signal
ORDER BY mix_adjusted_gap_pts DESC;
