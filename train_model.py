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
SEQUENCE_LENGTH = 10
PREDICTION_HORIZON = 2

def generate_dummy_data():
    """Generates dummy data with advanced features if the CSV is not found."""
    print("Generating dummy feature-engineered data for training...")
    n_samples = 200
    timestamps = pd.to_datetime(pd.date_range(start='2025-01-01', periods=n_samples, freq='0.5S'))

    base_rtt = np.random.rand(n_samples) * 0.05 + np.sin(np.arange(n_samples) / 10) * 0.02 + 0.01
    data = {
        'timestamp': timestamps,
        'length_sum': np.random.randint(10000, 50000, size=n_samples),
        'rtt_mean': base_rtt,
        'rtt_min': base_rtt * 0.9,
        'rtt_max': base_rtt * 1.1,
        'rtt_std': base_rtt * 0.05,
        'packet_count': np.random.randint(20, 150, size=n_samples),
        'rtt_mean_trend': np.random.randn(n_samples) * 0.001,
        'packet_count_trend': np.random.randn(n_samples) * 5,
        'length_sum_trend': np.random.randn(n_samples) * 1000,
        'rtt_volatility': base_rtt * 0.2
    }
    df = pd.DataFrame(data)
    df.to_csv(PROCESSED_CSV_FILE, index=False)
    print(f"Dummy data saved to {PROCESSED_CSV_FILE}")
    return df

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: '{filepath}' not found.")
        return generate_dummy_data()

    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def create_sequences(data, target, seq_length, horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i:(i + seq_length)])
        y.append(target[(i + seq_length):(i + seq_length + horizon)].mean())
    return np.array(X), np.array(y)

def build_model(input_shape):
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
    df = load_data(PROCESSED_CSV_FILE)

    # --- Use all engineered features for training ---
    features_to_use = [
        'length_sum', 'rtt_mean', 'rtt_min', 'rtt_max', 'rtt_std',
        'packet_count', 'rtt_mean_trend', 'packet_count_trend',
        'length_sum_trend', 'rtt_volatility'
    ]
    target_variable = 'rtt_mean' # We are still predicting the mean RTT

    # Ensure all features are present
    for col in features_to_use:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in the processed data. Please check pcap_parser.py.")
            return

    df_features = df[features_to_use]
    df_target = df[target_variable]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df_features)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler for {len(features_to_use)} features saved to {SCALER_FILE}")

    X, y = create_sequences(scaled_features, df_target.values, SEQUENCE_LENGTH, PREDICTION_HORIZON)

    if len(X) == 0:
        print("Not enough data to create sequences.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    print("Starting model training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
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
