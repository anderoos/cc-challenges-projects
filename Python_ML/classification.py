import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
import numpy as np

#================== Read in dataframe ==================
data = pd.read_csv('heart_failure.csv')
print(data.info())
# Preview label
print(data.death_event.value_counts(normalize=True))
# print(Counter(data.death_event))

# Data Extraction
y = data[['death_event']]
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

#================== Preprocessing ==================
# OHE and split
x = pd.get_dummies(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.3)

# Transform features
ct = ColumnTransformer([('numeric', StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])
x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)

# Encode labels to categorical
le = LabelEncoder()
y_train = le.fit_transform(y_train.astype(str))
y_test = le.transform(y_test.astype(str))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#================== Model Design ==================
# Build model
model = Sequential()
model.add(InputLayer(input_shape=x_train.shape[1],))
model.add(Dense(12, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Build callback
class myCallBack(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >0.95):
            self.model.stop_training = True
callback = myCallBack()

# Train and evaluate
model.fit(x_train, y_train, epochs=200, batch_size=16, callbacks=[callback], verbose=2)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

# Classification report
y_estimates = model.predict(x_test)
y_estimates = np.argmax(y_estimates, axis=1)
y_true = np.argmax(y_test, axis=1)

classification_report_results = classification_report(y_true, y_estimates)
print(classification_report_results)
