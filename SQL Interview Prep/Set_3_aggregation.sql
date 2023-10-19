-- Find the number of apps by genre.
-- Make sure to pass this step before attempting checkpoint 2.

SELECT genre, COUNT(*)
FROM apps
GROUP BY genre
ORDER BY 2 DESC;

-- Get the total number of reviews of all apps by genre.
-- Limit the results for genres where the total number of app reviews is over 30 million.

SELECT genre,
SUM(reviews)
FROM apps
GROUP BY genre
HAVING SUM(reviews) > 30000000
ORDER BY 2 DESC;