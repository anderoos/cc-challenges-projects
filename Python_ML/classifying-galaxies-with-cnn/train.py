import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

import app

# Preview dataset
input_data, labels = load_galaxy_data()
print(input_data.shape[1:4])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size=0.2, shuffle=True, random_state=222, stratify=labels)

# Image data generator
generator = ImageDataGenerator(rescale=1.0/255)
training_iter = generator.flow(X_train, y_train, batch_size=5)
validation_iter = generator.flow(X_test, y_test, batch_size=5)

# Build model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)

model = Sequential()
model.add(Input(shape=input_data.shape[1:4],))
model.add(Conv2D(8, 3, strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(8, 3, strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()], optimizer=optimizer)

model.summary()

# Fitting model
model.fit(training_iter, steps_per_epoch=len(X_train)/5, epochs=8, validation_data = validation_iter, validation_steps = len(X_test))

from visualize import visualize_activations
visualize_activations(model,validation_iter)
