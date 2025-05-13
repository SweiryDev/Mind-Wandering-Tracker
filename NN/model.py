import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

# Load input and output data
X = np.load("../data/AmitS_landmark.npy")
Y = np.load("../data/AmitS_distance.npy")

# Reshape input 
# X = X.reshape((X.shape[0], -1))

# Round output data to binary
middle_distance = np.max(Y) // 2
roundOut = np.vectorize(lambda t: 1 if (t < middle_distance) else 0)
Y = roundOut(Y)
Y = np.concat((np.split(Y, [30])[1], np.zeros(30)))

# Split the data
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, shuffle=False) 

# Define the model (layers test!)
model = tf.keras.Sequential([
    layers.Input(shape=(68,2)),
    layers.SimpleRNN(256, return_sequences=False),  # or use LSTM/GRU
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output a single scalar
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, batch_size=32)
