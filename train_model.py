import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib
import json

# --- Configuration ---
PROCESSED_CSV_FILE = 'processed_data.csv'
MODEL_FILE = 'congestion_model.keras'
SCALER_FILE = 'scaler.gz'
ACTIVITY_MAP_FILE = 'activity_map.json'
SEQUENCE_LENGTH = 10
PREDICTION_HORIZON = 2

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Processed data file '{filepath}' not found.")
        print("Please run pcap_parser.py first to generate the data.")
        return None

    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def create_sequences(data, categorical_data, rtt_data, loss_data, seq_length, horizon, rtt_threshold=0.20):
    X_ts, X_cat, y = [], [], []
    for i in range(len(data) - seq_length - horizon):
        # Define the future window
        future_window = (i + seq_length)
        future_end = i + seq_length + horizon

        # --- Define Label based on RTT increase OR Packet Loss ---
        label = 0 # Default to 'Stable'

        # Condition 1: Significant Packet Loss in the future
        future_packet_loss = loss_data[future_window:future_end].sum()
        if future_packet_loss > 0:
            label = 1

        # Condition 2: Significant RTT increase in the future
        else:
            current_rtt = rtt_data[i + seq_length - 1]
            future_rtt = rtt_data[future_window:future_end].mean()
            if current_rtt > 0.001 and (future_rtt - current_rtt) / current_rtt > rtt_threshold:
                label = 1

        y.append(label)
        X_ts.append(data[i:(i + seq_length)])
        X_cat.append(categorical_data[i + seq_length - 1])

    return np.array(X_ts), np.array(X_cat), np.array(y)

def build_multi_input_model(ts_shape, num_activities, embedding_dim=5):
    # --- Time-Series Input Branch ---
    ts_input = Input(shape=ts_shape, name='time_series_input')
    lstm_out = LSTM(50, return_sequences=True)(ts_input)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = LSTM(50, return_sequences=False)(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)

    # --- Categorical Input Branch ---
    cat_input = Input(shape=(1,), name='activity_type_input')
    embedding_out = Embedding(input_dim=num_activities, output_dim=embedding_dim, input_length=1)(cat_input)
    embedding_out = Flatten()(embedding_out) # Flatten the embedding output

    # --- Merged Branch ---
    merged = Concatenate()([lstm_out, embedding_out])
    merged = Dense(25, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    output = Dense(1, activation='sigmoid', name='output')(merged)

    model = Model(inputs=[ts_input, cat_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Multi-input model built successfully.")
    model.summary()
    return model

def main():
    df = load_data(PROCESSED_CSV_FILE)
    if df is None: return

    with open(ACTIVITY_MAP_FILE, 'r') as f:
        num_activities = len(json.load(f))

    features_to_use = [
        'length_sum', 'rtt_mean', 'rtt_min', 'rtt_max', 'rtt_std',
        'packet_count', 'lost_packet_count', 'rtt_mean_trend',
        'packet_count_trend', 'length_sum_trend', 'rtt_volatility'
    ]
    categorical_feature = 'activity_type'
    target_variable = 'rtt_mean'

    for col in features_to_use + [categorical_feature]:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found. Please re-run pcap_parser.py.")
            return

    df_features = df[features_to_use]
    df_cat = df[categorical_feature]
    df_rtt = df[target_variable]
    df_loss = df['lost_packet_count']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler for {len(features_to_use)} features saved.")

    X_ts, X_cat, y = create_sequences(scaled_features, df_cat.values, df_rtt.values, df_loss.values, SEQUENCE_LENGTH, PREDICTION_HORIZON)

    if len(X_ts) == 0:
        print("Not enough data to create sequences.")
        return

    # Split all three arrays together
    X_ts_train, X_ts_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_ts, X_cat, y, test_size=0.2, random_state=42
    )
    print(f"Data split into {len(X_ts_train)} training samples and {len(X_ts_test)} testing samples.")

    model = build_multi_input_model(
        ts_shape=(X_ts_train.shape[1], X_ts_train.shape[2]),
        num_activities=num_activities
    )

    print("Starting model training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Pass inputs as a list
    model.fit(
        [X_ts_train, X_cat_train], y_train,
        epochs=50,
        batch_size=32,
        validation_data=([X_ts_test, X_cat_test], y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    print("Model training complete.")
    loss, accuracy = model.evaluate([X_ts_test, X_cat_test], y_test)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == '__main__':
    main()
