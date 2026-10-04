-- Consumer segments and copy-paste ("template") complaints
SELECT 'Older American'            AS segment, COUNT(*) AS complaints, ROUND(100.0 * AVG(relief), 1) AS relief_rate_pct FROM complaints WHERE older_american = 1
UNION ALL
SELECT 'Servicemember',                         COUNT(*), ROUND(100.0 * AVG(relief), 1) FROM complaints WHERE servicemember = 1
UNION ALL
SELECT 'Untagged consumers',                    COUNT(*), ROUND(100.0 * AVG(relief), 1) FROM complaints WHERE older_american = 0 AND servicemember = 0
UNION ALL
SELECT 'Template text (filed 2+ times)',        COUNT(*), ROUND(100.0 * AVG(relief), 1) FROM complaints WHERE is_duplicate_text = 1
UNION ALL
SELECT 'Unique text',                           COUNT(*), ROUND(100.0 * AVG(relief), 1) FROM complaints WHERE is_duplicate_text = 0
UNION ALL
SELECT 'Template text filed 50+ times',         COUNT(*), ROUND(100.0 * AVG(relief), 1) FROM complaints WHERE dup_count >= 50;
