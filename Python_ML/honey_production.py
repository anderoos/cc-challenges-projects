import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Import
df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# Preview
# print(df.head())

# Aggregate
prod_per_year = df.groupby('year').agg({'totalprod': 'mean'}).reset_index()

# Extract and reshape
print(prod_per_year.head())
X = prod_per_year['year'].values.reshape(-1 ,1)
y = prod_per_year.totalprod.values.reshape(-1 ,1)

# Instantiate and fit
regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)
print(regr.intercept_)

# Create regression
y_pred = regr.predict(X)
plt.scatter(X, y)
plt.plot(X, y_pred)


# Create prediction to 2050
nums = np.arange(2013, 2051)
X_future= nums.reshape(-1 ,1)
print(X_future)

future_predict = regr.predict(X_future)
plt.plot(X_future, future_predict, color='red')
plt.show()
