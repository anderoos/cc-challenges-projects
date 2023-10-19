-- Given the final scores of several NBA games, use CASE to return the results for each game:

-- If home team won, return ‘HOME WIN’.
-- If away team won, return ‘AWAY WIN’.

SELECT id, 
  CASE 
    WHEN home_points > away_points THEN 'HOME WIN'
    ELSE 'AWAY WIN'
  END AS 'outcome'
FROM nba_matches;