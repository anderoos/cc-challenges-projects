-- Using a subquery, get all students in math who are also enrolled in english.
SELECT *
FROM math_students
WHERE student_id IN (
  SELECT student_id
  FROM english_students
);

-- Using a subquery, find out which students in math are in the same grade level as the student with id 7.
SELECT *
FROM math_students
WHERE grade IN (
  SELECT grade
  FROM math_students
  WHERE student_id = 7
);