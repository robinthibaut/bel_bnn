import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pandas as pd
from keras import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

fw = "/Users/rthibaut/PycharmProjects/bel_bnn/data/forward_RT.txt"
samples = "/Users/rthibaut/PycharmProjects/bel_bnn/data/samples_RT.txt"

forwards = np.loadtxt(fw, delimiter=',')
samples = np.loadtxt(samples, delimiter=',')

print(forwards.shape)
# (250000, 45)
# Time-domain TDEM data with 250000 samples and 45 channels.
# T/s or V/amp*m
# Time steps: logarithm spaced

print(samples.shape)
# (250000, 7)
# 7 features
# - altitude of the airborne EM system
# - thickness of first layer
# - thickness of a transition zone
# - thickness of a saltwater lens
# - conductivity of the first layer
# - conductivity of the saltwater lens
# - conductivity of the halfspace below the lens

gate_centers = np.r_[
                   -1.085, 0.415, 2.415, 4.415, 6.415, 8.415, 10.415, 12.915, 16.415, 20.915, 26.415, 33.415, 42.415,
                   53.915, 68.415, 86.415, 108.915, 136.915, 172.415, 217.915, 274.915, 346.915, 437.915, 551.915,
                   695.915, 877.415, 1105.915, 1394.415, 1758.415, 2216.915, 2794.915, 3523.915, 4442.915, 5601.415,
                   7061.415, 8902.415
               ] * 1e-6
gate_times_lm = gate_centers[3:26]
gate_times_hm = gate_centers[14:]
# Concatenate them to get the 45 gate times

# We want to predict "samples" from "forwards"

# We nees to reformat the target.
# Let's predict each pair (thickness, conductivity) separately, and skip the altitude.
# So we have 6 targets.
samples = samples[:, 1:]
samples = samples[:, [0, 3, 1, 4, 2, 5]]
samples = samples.reshape(-1, 3, 2)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(forwards, samples, test_size=0.2, random_state=42)

# Scaling the features
x_scaler = StandardScaler()
depth_scaler = StandardScaler()
conductivity_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_train[:, :, 0] = depth_scaler.fit_transform(y_train[:, :, 0])
y_test[:, :, 0] = depth_scaler.transform(y_test[:, :, 0])

y_train[:, :, 1] = conductivity_scaler.fit_transform(y_train[:, :, 1])
y_test[:, :, 1] = conductivity_scaler.transform(y_test[:, :, 1])

# Flatten the target
y_train = y_train.reshape(-1, 6)

# Reshape for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

output_dim = y_train.shape[1]  # number of outputs
hidden_units = 64  # number of hidden units in the dense layer
num_components = 3  # number of components in the mixture
params_size = tfp.layers.MixtureNormal.params_size(num_components, output_dim)  # size of the parameters

model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(params_size),
    tfp.layers.MixtureNormal(num_components, output_dim)
])


@tf.autograph.experimental.do_not_convert
def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=nll)

# Fit the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
history = model.fit(X_train, y_train, epochs=1000, validation_split=0.1, callbacks=[early_stopping], batch_size=32)

# Plot training history
plt.plot(history.history['loss'], label='train')
