-- Select the name, genre, and rating of apps in descending order of their rating, and limit the result to 20 rows.
SELECT name,
genre,
rating
FROM apps
ORDER BY rating DESC
LIMIT 20;