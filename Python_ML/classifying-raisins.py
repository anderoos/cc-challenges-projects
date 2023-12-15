<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

# # Classify Raisins with Hyperparameter Tuning Project
#
# - [View Solution Notebook](./solution.html)
# - [View Project Page](https://www.codecademy.com/projects/practice/mle-hyperparameter-tuning-project)

# ### 1. Explore the Dataset

# In[1]:


# 1. Setup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

raisins = pd.read_csv(&#39;Raisin_Dataset.csv&#39;)
raisins.head()


# In[4]:


# 2. Create predictor and target variables, X and y
raisins_target = raisins[[&#39;Class&#39;]]
raisins_predictor = raisins.drop(&#39;Class&#39;, axis=1)


# In[5]:


# 3. Examine the dataset
raisins_predictor.head()
raisins_target.head()


# In[6]:


# 4. Split the data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(raisins_predictor, raisins_target, random_state=49)


# ### 2. Grid Search with Decision Tree Classifier

# In[7]:


# 5. Create a Decision Tree model
tree = DecisionTreeClassifier()


# In[8]:


# 6. Dictionary of parameters for GridSearchCV
parameters = {
    &#34;max_depth&#34;: [3, 5, 7],
    &#34;min_samples_split&#34;: [2, 3, 4]
}


# In[10]:


# 7. Create a GridSearchCV model
grid = GridSearchCV(tree, parameters)

#Fit the GridSearchCV model to the training data
grid.fit(X_train, y_train)


# In[11]:


# 8. Print the model and hyperparameters obtained by GridSearchCV
print(grid.best_estimator_)

# Print best score
print(grid.best_score_)

# Print the accuracy of the final model on the test data
print(grid.score(X_test, y_test))


# In[16]:


# 9. Print a table summarizing the results of GridSearchCV
params_df = pd.DataFrame(grid.cv_results_[&#39;params&#39;])
score_df = pd.DataFrame(grid.cv_results_[&#39;mean_test_score&#39;], columns=[&#39;score&#39;])

results_df = pd.concat([params_df, score_df], axis=1)
results_df


# ### 2. Random Search with Logistic Regression

# In[17]:


# 10. The logistic regression model
lr = LogisticRegression(solver=&#39;liblinear&#39;, max_iter=1000)


# In[19]:


# 11. Define distributions to choose hyperparameters from
from scipy.stats import uniform

lr_params = {
    &#39;penalty&#39;: [&#39;l1&#39;, &#39;l2&#39;],
    &#39;C&#39;: uniform(loc=0, scale=100)
}


# In[21]:


# 12. Create a RandomizedSearchCV model
clf = RandomizedSearchCV(lr, lr_params, n_iter=10)

# Fit the random search model
clf.fit(X_train, y_train)


# In[31]:


# 13. Print best esimatore and best score
print(clf.best_estimator_)
print(clf.best_score_)

#Print a table summarizing the results of RandomSearchCV
clf_scores_df = pd.DataFrame(clf.cv_results_)

clf_scores_df[[&#39;param_C&#39;, &#39;param_penalty&#39;, &#39;mean_test_score&#39;]].sort_values(by=&#39;mean_test_score&#39;, ascending=False)


# In[ ]:




<script type="text/javascript" src="https://www.codecademy.com/assets/relay.js"></script></body></html>
