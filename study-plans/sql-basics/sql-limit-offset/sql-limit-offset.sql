-- Write your SQL query here
SELECT product, revenue, sale_date
FROM sales
ORDER BY revenue DESC, sale_date ASC
LIMIT 3 OFFSET 1;