# Dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

# Load in data and preview
life_expectancy_df = pd.read_csv('life_expectancy.csv')
# print(life_expectancy_df.info())

# Split into features and labels
labels = life_expectancy_df[['Life expectancy']]
features = life_expectancy_df.drop(['Country', 'Life expectancy'], axis=1)
# print(labels.info())
# print(features.info())

# Perform OHE on Categorical Variables
categorical_feature_names = ['Status']
status_ohe = pd.get_dummies(features['Status'])
features = pd.concat([features, status_ohe], axis=1)
features.drop(categorical_feature_names, axis=1, inplace=True)

# Preview features
# print(features.info())
# for col in features.columns:
#   print(features[col].describe())

# Ideally I would plot each feature here and observe the distributions of each.
# Here, I'll follow codecademy's instructions.

# Split into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=1, test_size=0.3)

# Standardize numeric features
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns

ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')
X_train_scaled = ct.fit_transform(X_train)
X_test_scaled = ct.transform(X_test)

# =================== Assessing baseline ===================
print(labels.describe())
dummy_reg = DummyRegressor(strategy='mean')
dummy_reg.fit(X_train_scaled, y_train)
y_baseline = dummy_reg.predict(X_test_scaled)
mae_baseline = mean_absolute_error(y_test, y_baseline)
print(mae_baseline)
# MAE baseline is 7.55

# =================== Building model ===================
model = Sequential()
input_layer = InputLayer(input_shape = (features.shape[1], ))
optimizer = Adam(learning_rate=0.1)

# Add input layer to model
# Working with regresssion (life_expectancy), one output neuron is sufficient
model.add(input_layer)
model.add(Dense(1))
model.compile(loss='mse', metrics=['mae'], optimizer=optimizer)

# Model Summary
model.summary()

# Training model
# Doubled the epochs from 40 to 80, implemnented callback
class myCallBack(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('mae') < 3):
            print('Terminating...')
            self.model.stop_training = True

callbacks = myCallBack()

model.fit(X_train_scaled, y_train, epochs= 40, batch_size=1, verbose=0, callbacks=[callbacks])
res_mse, res_mae = model.evaluate(X_train_scaled, y_train, verbose=0)

print('Model Evaluation:')
print(res_mse, res_mae)
# MAE of 3.27 on validation set!

# Evaluate model
y_test_prediction = model.predict(X_test_scaled)
print('Model Evaluation on test set:')
mae_test = mean_absolute_error(y_test, y_test_prediction)
print(mae_test)
# MAE of 3.39 on test set!

