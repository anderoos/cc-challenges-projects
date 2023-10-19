-- Given an orders table, calculate the price times quantity of each order. Include the id and product_id columns in the result.

SELECT id,
product_id,
price * quantity AS 'total'
FROM orders;