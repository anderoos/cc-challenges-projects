import codecademylib3_seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

# Explore data
# print(aaron_judge.head())
# print(aaron_judge.description.unique())
# print(aaron_judge.type.unique())
# print(aaron_judge.info())

# Transform type data to numeric
# aaron_judge['type'] = aaron_judge.type.map({'S': 1, 'B': 0})
#   # Drop NA values
# aaron_judge.dropna(inplace=True, subset=['type', 'plate_x', 'plate_z'])
# print(aaron_judge.type.unique())

# Initialize plot
# fig, ax = plt.subplots()
# ax.scatter(data=aaron_judge, x='plate_x', y='plate_z', c='type', cmap=plt.cm.coolwarm, alpha=0.25)
# ax.set_ylabel('plate_z')
# ax.set_xlabel('plate_x')

# Create SVC model with training and test sets
# training_set, testing_set = train_test_split(aaron_judge, random_state=1, test_size=0.2)

# Initialize SVC classifier, fit training data
# classifier = SVC(kernel = 'rbf', gamma=100, C=100)
# classifier.fit(training_set[['plate_x', 'plate_z']], training_set[['type']])

# Visualize boundary
# draw_boundary(ax, classifier)
# plt.show()
# plt.clf()

# Model accuracy
# print(classifier.score(testing_set[['plate_x', 'plate_z']], testing_set[['type']]))
# 0.8245; gamma='scale', C=1.0
# 0.7641; gamma=100, C=100

# Iterate for best score
# best_score = 0
# best_gamma = 0
# best_C = 0
# gamma_values = np.arange(1, 5, 1)
# C_values = np.arange(1, 5, 0.1)

# for i in gamma_values:
#   for j in C_values:
#     classifier = SVC(kernel = 'rbf', gamma=i, C=j)
#     classifier.fit(training_set[['plate_x', 'plate_z']], training_set[['type']])
#     score = classifier.score(testing_set[['plate_x', 'plate_z']], testing_set[['type']])

#     # Update values if value is superior
#     if score > best_score:
#       best_score = score
#       best_gamma = i
#       best_C = j

# print(f'Best Score: {best_score}')
# print(f'Best Gamma: {best_gamma}')
# print(f'Best C: {best_C}')
# Best score of 83.01% was found at gamma=1 and C=3

# Encapsulate code in function
def getStrikeZones(player):
  # Declaration
  parameter_dict = {
    'gamma': np.arange(1, 11, 1),
    "C": np.arange(1, 10, 0.5)
  }

  # Preprocessing
  player['type'] = player.type.map({'S': 1, 'B': 0})
  # Drop NA values
  player.dropna(inplace=True, subset=['type', 'plate_x', 'plate_z'])

  # Split train data
  training_set, testing_set = train_test_split(player, random_state=1, test_size=0.2)
  training_features = training_set[['plate_x', 'plate_z']]
  training_labels = training_set[['type']]
  testing_features = training_set[['plate_x', 'plate_z']]
  testing_labels = training_set[['type']]

  # Classifier with GridSearchCV
  classifier = SVC(kernel='rbf')
  grid_search = GridSearchCV(classifier, parameter_dict, cv=5, refit=True)

  # Fit and retrieve best params
  grid_search.fit(training_features, training_labels)
  best_params = grid_search.best_params_
  best_score = grid_search.best_score_
  best_estimator = grid_search.best_estimator_

  # Tuned classifier
  tuned_classifier = CSV(kerenel='rbf', C=best_params['C'], gamma=best_params['gamma'])
  tuned_classifier.fit(training_features, training_labels)

  # Plot tuned classifier
  fig, ax = plt.subplots()
  ax.scatter(data=aaron_judge, x='plate_x', y='plate_z', c='type', cmap=plt.cm.coolwarm, alpha=0.25)
  ax.set_ylabel('plate_z')
  ax.set_xlabel('plate_x')
  ax.set_ylim(-2, 6)
  ax.set_xlim(-3, 3)
  draw_boundary(ax, classifier)
  plt.show()
  plt.clf()

  print(f'Best Score: {best_score}')
  print(f'Best Parameters: {best_params}')
  print(f'Best Estimator: {best_estimator}')

getStrikeZones(aaron_judge)
