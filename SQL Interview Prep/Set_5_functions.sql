-- Find the lowest and highest rating for all apps using two different queries.

SELECT MIN(rating) AS lowest_rating
FROM apps;
SELECT MAX(rating) AS highest_rating
FROM apps;

-- Get the average rating of all apps, rounded to 2 decimal places. Alias the result column as ‘average rating’.

SELECT ROUND(AVG(rating), 2) AS 'average rating'
FROM apps;
