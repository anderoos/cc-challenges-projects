import pandas as pd
import numpy as np
import codecademylib3
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

path_to_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

col_names = [
    'age', 'workclass', 'fnlwgt','education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain','capital-loss',
    'hours-per-week','native-country', 'income'
]

df = pd.read_csv(path_to_data, header=None, names = col_names)
print(df.head())

#Clean columns by stripping extra whitespace for columns of type "object"
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].str.strip()

target_column = "income"
raw_feature_cols = [
    'age',
    'education-num',
    'workclass',
    'hours-per-week',
    'sex',
    'race'
]

##1. Percentage of samples with income < and > 50k
print(df.income.value_counts(normalize=True))

##2. Data types of features
print(df.info())

##3. Preparing the features
X = pd.get_dummies(df[raw_feature_cols], drop_first=True)

##4. Convert target variable to binary
y = np.where(df['income'] == '<=50K', 0, 1)

##5a. Create train-est split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
##5b. Create base estimator and store it as decision_stump
decision_stump = DecisionTreeClassifier(max_depth = 1)

##6. Create AdaBoost Classifier
ada_classifier = AdaBoostClassifier(base_estimator=decision_stump)

##7. Create GradientBoost Classifier
grad_classifier = GradientBoostingClassifier()

##8a.Fit models and get predictions
ada_classifier.fit(X_train, y_train)
y_pred_ada = ada_classifier.predict(X_test)
grad_classifier.fit(X_train, y_train)
y_pred_grad = grad_classifier.predict(X_test)

##8b. Print accuracy and F1
print(f"AdaBoost accuracy: {accuracy_score(y_test, y_pred_ada)}")
print(f"AdaBoost f1-score: {f1_score(y_test, y_pred_ada)}")
print(f"Gradient Boost accuracy: {accuracy_score(y_test, y_pred_grad)}")
print(f"Gradient Boost f1-score: {f1_score(y_test, y_pred_grad)}")

##9. Hyperparameter Tuning
n_estimators_list = [10, 30, 50, 70, 90]
from sklearn.model_selection import GridSearchCV

estimator_params = {
  'n_estimators': n_estimators_list
}

ada_gridsearch = GridSearchCV(ada_classifier, estimator_params, cv=5, scoring='accuracy', verbose=True)
ada_gridsearch.fit(X_train, y_train)
y_pred_ada_grid = ada_gridsearch.predict(X_test)
print(''' GRIDSEARCHCV ''')
print(f"AdaBoost accuracy: {accuracy_score(y_test, y_pred_ada_grid)}")
print(f"AdaBoost f1-score: {f1_score(y_test, y_pred_ada_grid)}")
print(f"AdaBoost Params: {ada_gridsearch.best_params_}")

##10. Plot mean test scores
ada_scores_list = ada_gridsearch.cv_results_['mean_test_score']
plt.scatter(n_estimators_list, ada_scores_list)
plt.show()
plt.clf()
