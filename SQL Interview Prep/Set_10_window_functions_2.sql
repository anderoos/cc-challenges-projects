-- Write a query using a window function with ROW_NUMBER and ORDER BY to see where each row falls in the amount of gross.

SELECT 
ROW_NUMBER() OVER (
  ORDER BY gross 
  ) as 'row_index',
title,
week,
gross
FROM box_office;