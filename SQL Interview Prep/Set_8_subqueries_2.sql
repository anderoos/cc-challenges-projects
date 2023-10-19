-- Using a subquery, find all students enrolled in english class who are not also enrolled in math class.

SELECT *
FROM english_students
WHERE student_id NOT IN (
  SELECT student_id
  FROM math_students
);

-- Using a subquery, find out what grade levels are represented in both the math and english classes.

SELECT grade
FROM math_students 
WHERE EXISTS (
  SELECT grade
  FROM english_students
);