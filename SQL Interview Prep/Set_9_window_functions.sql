-- Using a window function with PARTITION BY, get the running total in gross for each movie up to the current week and display it next to the current week column along with the title, week, and gross columns.

SELECT title,
week,
gross,
SUM(gross) OVER (
  PARTITION BY title
  ORDER BY week
) AS 'running_gross'
FROM box_office;