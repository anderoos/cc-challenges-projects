import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Data
df = pd.read_csv('wine_quality.csv')
y = df['quality']
features = df.drop(columns = ['quality'])


## 1. Data transformation
from sklearn.preprocessing import StandardScaler
# Zscore normalization, data transformation
standard_scaler_fitted = StandardScaler().fit(features)
X = standard_scaler_fitted.transform(features)


## 2. Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 99, test_size=0.2)

## 3. Fit a logistic regression classifier without regularization
from sklearn.linear_model import LogisticRegression

clf_nr = LogisticRegression(penalty='none')
clf_nr.fit(X_train, y_train)

## 4. Plot the coefficients
predictors = features.columns
coefficients = clf_nr.coef_.ravel()
coef = pd.Series(coefficients, predictors)
coef.plot(kind='bar', title = 'Coefficients (no regularization)')
plt.tight_layout()
plt.show()
plt.clf()

## 5. Training and test performance
from sklearn.metrics import f1_score
y_pred_test = clf_nr.predict(X_test)
y_pred_train = clf_nr.predict(X_train)
print('Training Score', f1_score(y_train, y_pred_train))
print('Testing Score', f1_score(y_test, y_pred_test))


## 6. Default Implementation (L2-regularized!)
clf_default = LogisticRegression()
clf_default.fit(X_train, y_train)

## 7. Ridge Scores
y_pred_train = clf_default.predict(X_train)
y_pred_test = clf_default.predict(X_test)

print('Ridge-regularized Training Score', f1_score(y_train, y_pred_train))
print('Ridge-regularized Testing Score', f1_score(y_test, y_pred_test))

## 8. Coarse-grained hyperparameter tuning
training_array = []
test_array = []
C_array = [0.0001, 0.001, 0.01, 0.1, 1]

for x in C_array:
    clf = LogisticRegression(C = x )
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    training_array.append(f1_score(y_train, y_pred_train))
    test_array.append(f1_score(y_test, y_pred_test))

## 9. Plot training and test scores as a function of C
plt.plot(C_array,training_array)
plt.plot(C_array,test_array)
plt.xscale('log')
plt.show()
plt.clf()

## 10. Making a parameter grid for GridSearchCV
C_array = np.logspace(-4, 0, 100)
param_grid = {
  "C": C_array
}

## 11. Implementing GridSearchCV with l2 penalty
from sklearn.model_selection import GridSearchCV
clf_gs = LogisticRegression()
gs = GridSearchCV(clf_gs, param_grid=param_grid, scoring="f1", cv=5)
gs.fit(X_train, y_train)

## 12. Optimal C value and the score corresponding to it
print(f'Best parameters: {gs.best_params_}')
print(f'Best training score: {gs.best_score_}')

## 13. Validating the "best classifier"
best_classifier = gs.best_params_['C']
clf_best = LogisticRegression(C = best_classifier)
clf_best.fit(X_train,y_train)
y_pred_best = clf_best.predict(X_test)
print(f'Best Training Score: {f1_score(y_test,y_pred_best)}')

## 14. Implement L1 hyperparameter tuning with LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV
C_array = np.logspace(-2,2,100)
clf_l1 = LogisticRegressionCV(Cs=C_array, cv = 5, penalty = 'l1', scoring = 'f1', solver = 'liblinear')
clf_l1.fit(X,y)

## 15. Optimal C value and corresponding coefficients
print(f'Best C value with L1: {clf_l1.C_}')
print(f'Coefficient with L1: {clf_l1.coef_}')
# L1 eliminated 'density' feature fromm this model.

## 16. Plotting the tuned L1 coefficients
coefficients = clf_l1.coef_.ravel()
coef = pd.Series(coefficients,predictors).sort_values()

plt.figure(figsize = (12,8))
coef.plot(kind='bar', title = 'Coefficients for tuned L1')
plt.tight_layout()
plt.show()
plt.clf()

