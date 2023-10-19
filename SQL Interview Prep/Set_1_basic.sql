-- Select the title, author, and average_rating of each book with an average_rating between 3.5 and 4.5.

SELECT title, 
author,
average_rating
FROM books
WHERE average_rating BETWEEN 3.5 AND 4.5;

-- Select all the unique authors from the table.

SELECT DISTINCT author,
COUNT(author)
FROM books
GROUP BY author
ORDER BY 2 COUNT(author);