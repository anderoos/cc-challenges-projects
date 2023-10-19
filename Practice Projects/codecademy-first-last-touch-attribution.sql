/*
Here's the first-touch query, in case you need it
*/

-- WITH first_touch AS (
--     SELECT user_id,
--         MIN(timestamp) as first_touch_at
--     FROM page_visits
--     GROUP BY user_id)
-- SELECT ft.user_id,
--     ft.first_touch_at,
--     pv.utm_source,
-- 		pv.utm_campaign
-- FROM first_touch ft
-- JOIN page_visits pv
--     ON ft.user_id = pv.user_id
--     AND ft.first_touch_at = pv.timestamp;

-- Task 1 / 2
SELECT utm_campaign, 
   COUNT(utm_campaign)
FROM page_visits 
GROUP BY utm_campaign;

SELECT DISTINCT utm_source, 
   utm_campaign
FROM page_visits;

SELECT DISTINCT page_name
FROM page_visits;

-- Task 3
WITH last_touch AS
( 
  SELECT user_id,
  max(timestamp) AS last_touch_time,
  utm_campaign
  FROM page_visits
  WHERE page_name = '4 - purchase'
  GROUP BY user_id
),
last_touch_table AS (
  SELECT lt.user_id,
    lt.last_touch_time,
    pv.utm_source,
    pv.utm_campaign,
    pv.page_name
  FROM last_touch lt
  JOIN page_visits pv 
  ON lt.user_id = pv.user_id
    AND lt.last_touch_time = pv.timestamp
  )
SELECT last_touch_table.utm_source,
    last_touch_table.utm_campaign,
    last_touch_table.page_name,
  COUNT(*) 
  FROM last_touch_table
  GROUP BY 1
  ORDER BY 3 DESC;

SELECT COUNT(*) AS visitor_count,
  page_name
  FROM page_visits
  GROUP BY page_name;
