import seaborn
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import codecademylib3

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head())
print(transactions.info())

# How many fraudulent transactions?
fraudulent_count = transactions.isFraud.value_counts()
print(fraudulent_count)

# Summary statistics on amount column
print(transactions.amount.describe())

# Create isPayment field
transactions['isPayment'] = transactions['type'].apply(lambda x: 1 if x == 'PAYMENT' or x == 'DEBIT' else 0)
print(transactions)

# More efficient
transactions['isPayment'] = ((transactions['type'] == 'PAYMENT') | (transactions['type'] == 'DEBIT')).astype(int)

# Create isMovement field
transactions['isMovement'] = ((transactions['type'] == 'CASHOUT')| (transactions['type'] == 'TRANSFER')).astype(int)

# Create accountDiff field
transactions['accountDiff'] = transactions['oldbalanceOrg'] - transactions['oldbalanceDest'].abs()


# Create features and label variables
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]
label = transactions['isFraud']

print(features)
print(label)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size = 0.3, random_state = 42)

# Normalize the features variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the model to the training data
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

# Score the model on the training data
print(model.score(X_train_scaled, Y_train))

# Score the model on the test data
print(model.score(X_test_scaled, Y_test))

# Print the model coefficients
print(model.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
new_transactions = np.array([transaction1, transaction2, transaction3])

# Combine new transactions into a single array
# see above

# Normalize the new transactions
new_transactions_scaled = scaler.fit_transform(new_transactions)

# Predict fraud on the new transactions
predict_fraud = model.predict(new_transactions_scaled)
print(predict_fraud)

# Show probabilities on the new transactions
print(model.predict_proba(new_transactions_scaled))

# Get Values