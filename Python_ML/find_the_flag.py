import codecademylib3
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name', 'landmass', 'zone', 'area', 'population', 'language', 'religion', 'bars', 'stripes', 'colours',
        'red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 'mainhue', 'circles',
        'crosses', 'saltires', 'quarters', 'sunstars', 'crescent', 'triangle', 'icon', 'animate', 'text', 'topleft',
        'botright']

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names=cols)

# variable names to use as predictors
var = ['red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 'mainhue', 'bars', 'stripes', 'circles', 'crosses',
       'saltires', 'quarters', 'sunstars', 'triangle', 'animate']

# Landmass dictionary
landmass_dict = {
    "N.America": 1,
    "S.America": 2,
    "Europe": 3,
    "Africa": 4,
    "Asia": 5,
    "Oceania": 6
}

target_continents = [landmass_dict['Asia'], landmass_dict['Europe']]

# Print number of countries by landmass, or continent
# print(df['landmass'].value_counts())

# Create a new dataframe with only flags from Europe and Oceania

df_36 = df[df['landmass'].isin(target_continents)]
print(df_36)

# Print the average values of the predictors for Europe and Oceania
df_36_agg = df_36.groupby('landmass')[var].mean()
print(df_36_agg)

# Create labels for only Europe and Oceania
labels = (df['landmass'].isin(target_continents)) * 1
print(labels)

# Print the variable types for the predictors
print(df_36[var].dtypes)

# Create dummy variables for categorical predictors
data = pd.get_dummies(df[var])
print(data)
# Split data into a train and test set
X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=5, test_size=0.4)
post_split = [X_train, X_test, y_train, y_test]
print([len(x) for x in post_split])

# Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21, 1)
acc_depth = []
best_depth = [0, 0]
for i in depths:
    model = DecisionTreeClassifier(max_depth=i)
    model.fit(X_train, y_train)
    model_score = model.score(X_test, y_test)
    acc_depth.append(model_score)
    if model_score > best_depth[1]:
        best_depth = [i, model_score]

# Plot the accuracy vs depth
plt.plot(depths, acc_depth)
plt.show()
plt.close()

# Find the largest accuracy and the depth this occurs
print(best_depth)

# Refit decision tree model with the highest accuracy and plot the decision tree
model = DecisionTreeClassifier(max_depth=best_depth[0])
model.fit(X_train, y_train)
plt.figure(figsize=(10, 8))
tree.plot_tree(model, feature_names=X_train.columns, max_depth=best_depth[0])
plt.tight_layout()
plt.show()
plt.close()

# Create a new list for the accuracy values of a pruned decision tree.  Loop through
# the values of ccp and append the scores to the list
ccp = np.arange(0, 0.1, 0.001)
acc_pruned = []
best_ccp = [0.0, 0.0]
for i in ccp:
    model = DecisionTreeClassifier(max_depth=best_depth[0], ccp_alpha=i)
    model.fit(X_train, y_train)
    model_score = model.score(X_test, y_test)
    acc_pruned.append(model_score)
    if model_score > best_ccp[1]:
        best_ccp = [i, model_score]

# Plot the accuracy vs ccp_alpha
plt.plot(ccp, acc_pruned)
plt.show()
plt.close()

# Find the largest accuracy and the ccp value this occurs
print(best_ccp)

# Fit a decision tree model with the values for max_depth and ccp_alpha found above
model = DecisionTreeClassifier(max_depth=best_depth[0], ccp_alpha=best_ccp[0])
model.fit(X_train, y_train)

# Plot the final decision tree
plt.figure(figsize=(20, 16))
tree.plot_tree(model, feature_names=X_train.columns, max_depth=best_depth[0],
               class_names=[f'not within targets', f'within targets'], label='all', filled=True)
plt.tight_layout()
plt.show()
plt.close()
