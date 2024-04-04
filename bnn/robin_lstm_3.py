import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

cwd = os.getcwd()
data = pd.read_pickle(os.path.join(cwd, "bnn", 'datags.pkl'))

# Assuming `data` is a DataFrame or a 2D NumPy array with shape [samples, features]
time_steps = 30  # Past time steps you're using to predict the next step
# Adjust X to use time_steps for predicting the next single step
X = []
y = []
for i in range(len(data) - time_steps - 1):  # minus 1 to ensure y[i] is valid
    X.append(data[i:(i + time_steps), :])
    y.append(data[i + time_steps, :])  # Assuming you want to predict all features at the next step

X = np.array(X)
y = np.array(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Reshape data for scaling
nsamples_train, nx_train, ny_train = X_train.shape
X_train_reshaped = X_train.reshape((nsamples_train, nx_train * ny_train))
X_train_scaled = scaler_X.fit_transform(X_train_reshaped)

nsamples, nx, ny = X_test.shape
X_test_reshaped = X_test.reshape((nsamples, nx * ny))
X_test_scaled = scaler_X.transform(X_test_reshaped)

# Scale y
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Reshape back to [samples, time steps, features] for LSTM input
X_train = X_train_scaled.reshape((nsamples_train, nx_train, ny_train))
X_test = X_test_scaled.reshape((nsamples, nx, ny))

# Define the LSTM model for single-step prediction
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(time_steps, y.shape[1]), dropout=0.2),
    Dense(y.shape[1])  # Output layer to predict all features at the next step
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train_scaled, epochs=100, validation_split=0.1, callbacks=[early_stopping],
                    batch_size=32)

# Plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


def predict_sequence(model, initial_sequence, n_steps, scaler_X, scaler_y):
    sequence = np.array(initial_sequence)
    predictions = []

    for _ in range(n_steps):
        # Ensure the sequence is the right shape [samples, time_steps, n_features]
        input_seq = sequence[-time_steps:].reshape((1, time_steps, -1))

        # Flatten the input sequence to 2D for scaling, then reshape back to 3D
        input_seq_flattened = input_seq.reshape(1, -1)  # Flatten to 2D
        input_seq_scaled = scaler_X.transform(input_seq_flattened)  # Scale the input
        input_seq_scaled = input_seq_scaled.reshape(1, time_steps, -1)  # Reshape back to 3D

        # Predict the next time step (scaled)
        pred_scaled = model.predict(input_seq_scaled)

        # Inverse transform the scaled prediction
        pred = scaler_y.inverse_transform(pred_scaled)

        # Append the prediction (ensure it's flattened to match the sequence format)
        predictions.append(pred.flatten())

        # Update the sequence with the predicted value (need to append in the original scale)
        sequence = np.append(sequence, pred, axis=0)

    return predictions


# Preparing an initial sequence from X_test (unscaled, original data)
# You should start with the original, unscaled data for the initial sequence
initial_sequence_index = 0  # Just an example index
initial_sequence = X[
                   initial_sequence_index:initial_sequence_index + time_steps]  # Get the sequence from the original dataset

# Transform the initial sequence with the scaler before prediction
initial_sequence_transformed = scaler_X.transform(initial_sequence.reshape(-1, y.shape[1])).reshape(1, time_steps, -1)

# Predict the next 30 steps based on the initial sequence
predicted_sequence = predict_sequence(model, initial_sequence_transformed, 30, scaler_X, scaler_y)
