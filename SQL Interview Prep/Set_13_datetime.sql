-- After a purchase is created, it can be returned within 7 days for a full refund.
-- Using modifiers, get the date of each purchase offset by 7 days in the future.

SELECT *,
DATE(purchase_date, '+7 days') AS 'return window end'
FROM purchases;

-- Get the hour that each purchase was made.
-- Which hour had the most purchases made?
SELECT 
STRFTIME('%H', purchase_date) AS 'hour_purchased',
COUNT(STRFTIME('%H', purchase_date)) AS 'num_purchased'
FROM purchases
GROUP BY hour_purchased
ORDER BY num_purchased DESC;