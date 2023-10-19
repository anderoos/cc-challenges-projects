-- Utilize CAST to calculate the average of the low and high temperatures for each date such that the result is of type REAL.
-- Select the date column and alias this result column as ‘average’.

SELECT date, 
CAST((high + low )AS REAL)/ 2.0 AS 'average'
FROM weather;
