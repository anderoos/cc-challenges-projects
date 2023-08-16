import codecademylib
import numpy as np

# Checkpoint 2
calorie_stats = np.genfromtxt('cereal.csv', delimiter=',')
# Checkpoint 3
calorie_stats_avg = np.mean(calorie_stats)
average_calories = calorie_stats_avg - 60
print(calorie_stats_avg)
# Checkpoint 4
calorie_stats_sorted = np.sort(calorie_stats)
# print(calorie_stats_sorted)
# Checkpoint 5
median_calories = np.median(calorie_stats)
print(median_calories)
# Checkpoint 6
for num in range(8):
  n_percentile = np.percentile(calorie_stats, num)
  if n_percentile >= 60:
    print(num, ":", n_percentile)
    break
nth_percentile = 70.0
# Checkpoint 7
more_calories = np.mean(calorie_stats > 60)
print(more_calories)
# Checkpoint 8
calorie_std = np.std(calorie_stats)
print(calorie_std)
# Checkpoint 9
# Comments:
# cereal.csv shows that 96% of different cereal brands on the market have over 60 calories per serving. 
# Median calories of cereal in this dataset is 110
# Positioning CrunchMunchies in the 4th percentile; significantly below the median.
# CrunchieMunchies can be marketed as a delicious low-calorie cereal