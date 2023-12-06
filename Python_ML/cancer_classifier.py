import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

# Load in data
breast_cancer_data = load_breast_cancer()

# Preview
print(breast_cancer_data.feature_names)
# print(breast_cancer_data.data[0])
# print(breast_cancer_data.target)
# print(breast_cancer_data.target_names)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)
all_sets = [X_train, X_test, y_train, y_test]

# Confirm that split has performed correctly
print([len(x) for x in all_sets])

# Instantiate fit and score KNClassifier with k=3
# classifier = KNeighborsClassifier(n_neighbors = 3)
# classifier.fit(X_train, y_train)
# classifier.score(X_test, y_test)
# R2 value of 0.974!

# Determine best k-value
k_scores = []
best_k = [0, 0]
for i in range(1, 101, 1):
  classifier = KNeighborsClassifier(n_neighbors = i)
  classifier.fit(X_train, y_train)
  classifier_score = classifier.score(X_test, y_test)
  k_scores.append(classifier_score)
  if classifier_score > best_k[1]:
    best_k = [i, classifier_score]

# Visualize scores
k_values = range(1, 101, 1)
fig, ax = plt.subplots()
ax.plot(k_values, k_scores)
ax.set_title("Breast Cancer Classifier Accuray", fontsize=18)
ax.set_xlabel("K values")
ax.set_ylabel("Validation Accuracy")
plt.show()
plt.close()
print(best_k)

# Classify with new k-score
classifier = KNeighborsClassifier(n_neighbors =23)
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)
y_predictions = classifier.predict(X_test)
  
