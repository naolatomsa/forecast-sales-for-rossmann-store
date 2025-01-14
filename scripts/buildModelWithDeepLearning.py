import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import numpy as np

def create_lstm_data(data, target_col, look_back=30):

    X, y = [], []
    # Ensure data is numpy array
    data = np.array(data)
    for i in range(len(data) - look_back):
        # Create sequences for features (X) and corresponding target (y)
        X.append(data[i:i + look_back, :])  # All features for the current sequence
        y.append(data[i + look_back, target_col])  # Target is the value in the target column for the next timestep
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2):

    model = Sequential([
        LSTM(lstm_units, activation='relu', input_shape=input_shape),
        Dropout(dropout_rate),
        Dense(1)  # Output layer for regression (single value)
    ])
    model.compile(optimizer='adam', loss='mse')  # Mean squared error for regression tasks
    return model

def train_lstm_model(model, X, y, epochs=10, batch_size=32, validation_split=0.2):
 
    print(f"Training LSTM model for {epochs} epochs...")
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    print("Training complete!")
    return history

def save_lstm_model(model, filename="lstm_model.h5"):

    model.save(filename)
    print(f"LSTM model saved as {filename}")

    
# # Example: Reusing the functions in your workflow
# scaled_train_data = trainData.values  # Assuming trainData is already scaled
# look_back = 30

# # Step 1: Prepare data
# X_lstm, y_lstm = create_lstm_data(scaled_train_data, target_col=trainData.columns.get_loc('Sales'), look_back=look_back)

# # Step 2: Build the model
# input_shape = (look_back, X_lstm.shape[2])
# lstm_model = build_lstm_model(input_shape)

# # Step 3: Train the model
# train_lstm_model(lstm_model, X_lstm, y_lstm, epochs=10, batch_size=32, validation_split=0.2)

# # Step 4: Save the model
# save_lstm_model(lstm_model, filename="sales_forecasting_lstm_model.h5")
