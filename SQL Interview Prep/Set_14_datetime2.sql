-- Using string formatting and substitutions, get the month and day for each purchase in the form ‘mm-dd’.
-- Give this new column a name of ‘reformatted’.

SELECT *,
STRFTIME('%m-%d', purchase_date) AS 'reformatted'
FROM purchases;