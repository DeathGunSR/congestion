import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib

# --- Configuration ---
PROCESSED_CSV_FILE = 'processed_data.csv'
MODEL_FILE = 'congestion_model.keras'
SCALER_FILE = 'scaler.gz'
SEQUENCE_LENGTH = 2 # 2 * 0.5s = 1 second of past data
PREDICTION_HORIZON = 10 # 10 * 0.5s = 5 seconds of future prediction

def generate_dummy_data():
    """Generates dummy aggregated data if the processed CSV is not found."""
    print("Generating dummy aggregated data for training...")
    n_samples = 200 # Fewer samples, as each represents a 5s interval

    # Create a datetime index with 0.5-second frequency
    timestamps = pd.to_datetime(pd.date_range(start='2025-01-01', periods=n_samples, freq='0.5S'))

    data = {
        'timestamp': timestamps,
        'length': np.random.randint(10000, 50000, size=n_samples), # Sum of lengths
        'rtt': np.random.rand(n_samples) * 0.1 + np.sin(np.arange(n_samples) / 10) * 0.05 + 0.01,
        'packet_count': np.random.randint(50, 300, size=n_samples)
    }
    df = pd.DataFrame(data)
    df.to_csv(PROCESSED_CSV_FILE, index=False)
    print(f"Dummy data saved to {PROCESSED_CSV_FILE}")
    return df

def load_data(filepath):
    """Loads data from the CSV file."""
    if not os.path.exists(filepath):
        print(f"Warning: '{filepath}' not found.")
        return generate_dummy_data()

    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def create_sequences(data, target, seq_length, horizon):
    """Creates input sequences and corresponding target values."""
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i:(i + seq_length)])
        # Target is the average RTT in the prediction horizon
        y.append(target[(i + seq_length):(i + seq_length + horizon)].mean())
    return np.array(X), np.array(y)

def build_model(input_shape):
    """Builds, compiles, and returns the LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Model built successfully.")
    model.summary()
    return model

def main():
    """Main function to run the training pipeline."""
    df = load_data(PROCESSED_CSV_FILE)

    features_to_use = ['length', 'rtt', 'packet_count']
    target_variable = 'rtt'

    df_features = df[features_to_use]
    df_target = df[target_variable]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df_features)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler saved to {SCALER_FILE}")

    X, y = create_sequences(scaled_features, df_target.values, SEQUENCE_LENGTH, PREDICTION_HORIZON)

    if len(X) == 0:
        print("Not enough data to create sequences. Please use a larger pcap file or a shorter sequence length.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    print("Starting model training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    print("Model training complete.")
    loss = model.evaluate(X_test, y_test)
    print(f"\nTest Loss (MSE): {loss}")

    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == '__main__':
    main()
