-- Write your SQL query here
SELECT p.name, p.price,
       ROUND(p.price - (SELECT AVG(price) FROM products), 2) AS vs_avg
FROM products p
WHERE p.id IN (SELECT product_id FROM sales)
ORDER BY vs_avg DESC, p.name ASC;
