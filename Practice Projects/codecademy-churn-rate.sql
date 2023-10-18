-- 1 . Data Preview 
-- SELECT * FROM subscriptions LIMIT 100;

-- 2 . Range 
-- SELECT min(subscription_start), max(subscription_start), min(subscription_end), max(subscription_end) FROM subscriptions LIMIT 100;
-- 2016/12/01 - 2017/03/30
-- 2017/01/01 - 2017/03/31

-- 3 . Churn rate for each segment
WITH months AS 
(
  SELECT
    '2017-01-01' AS first_day,
    '2017-01-31' AS last_day
  UNION
  SELECT
    '2017-02-01' AS first_day,
    '2017-02-28' AS last_day
  UNION
  SELECT
    '2017-03-01' AS first_day,
    '2017-03-31' AS last_day
),
-- CROSSTABLE CTE
crosstab AS
(
  SELECT *
  FROM subscriptions
  CROSS JOIN months
),
-- STATUS CTE, BY SEGMENT
status AS 
(
  SELECT id, first_day AS month,
  CASE 
    WHEN (subscription_start < first_day)
      AND ( subscription_end > first_day
        OR subscription_end IS NULL)
      AND (segment = 87) THEN 1
    ELSE 0
  END AS is_active_87,
  CASE
    WHEN (subscription_start < first_day)
      AND ( subscription_end > first_day
        OR subscription_end IS NULL)
      AND (segment = 30) THEN 1
    ELSE 0
  END AS is_active_30,
-- CANCELED SEGMENTS
  CASE 
    WHEN (subscription_end BETWEEN first_day AND last_day)
      AND (segment = 87) THEN 1
    ELSE 0 
  END AS is_canceled_87,
  CASE 
    WHEN (subscription_end BETWEEN first_day AND last_day)
     AND (segment = 30) THEN 1
   ELSE 0
  END AS is_canceled_30
  FROM crosstab
  ),
status_aggregate AS
(
  SELECT month,
  SUM(is_active_87) AS sum_active_87,
  SUM(is_active_30) AS sum_active_30,
  SUM(is_canceled_87) AS sum_canceled_87,
  SUM(is_canceled_30) AS sum_cancaled_30
  FROM status
  GROUP BY month
)

SELECT 
  month,
  (1.0 * sum_canceled_87/ sum_active_87) AS churn_87,
  (1.0 * sum_cancaled_30/ sum_active_30) AS churn_30
FROM status_aggregate;