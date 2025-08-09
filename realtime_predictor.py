import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import argparse
import os
import sys
import time
from collections import deque
from scapy.all import sniff
from feature_extractor import process_packets

# --- Configuration ---
MODEL_FILE = 'congestion_model.keras'
SCALER_FILE = 'scaler.gz'
SEQUENCE_LENGTH = 12  # Must match the training sequence length
TIME_INTERVAL_SECONDS = 5
PREDICTION_HORIZON = 1

# --- Prediction Thresholds ---
SEVERE_CONGESTION_THRESHOLD = 0.1
MODERATE_CONGESTION_THRESHOLD = 0.05

def get_network_interface():
    """Helper function to get a list of network interfaces."""
    try:
        from scapy.arch import get_windows_if_list
        interfaces = get_windows_if_list()
        return [iface['name'] for iface in interfaces]
    except ImportError:
        # For Linux/macOS
        return os.listdir('/sys/class/net/')

def main():
    """Main function to run the real-time prediction pipeline."""
    parser = argparse.ArgumentParser(description="Predict network congestion in real-time.")
    parser.add_argument("--ip", required=True, help="The IP address of this machine.")
    parser.add_argument("--iface", required=False, help="The network interface to sniff on (e.g., wlan0).")
    args = parser.parse_args()

    if not args.iface:
        print("No network interface specified. Available interfaces:")
        for iface in get_network_interface():
            print(f" - {iface}")
        print("\nPlease run the script again with the --iface argument.")
        return

    # 1. Load Model and Scaler
    print("Loading trained model and scaler...")
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    except (IOError, ImportError) as e:
        print(f"Error loading model or scaler: {e}\nPlease run train_model.py first.")
        return

    # 2. Initialize data queue
    # This deque will store the last SEQUENCE_LENGTH time intervals of data
    data_queue = deque(maxlen=SEQUENCE_LENGTH)
    features_to_use = ['length', 'rtt', 'packet_count']

    print(f"\nStarting real-time congestion prediction on interface '{args.iface}'...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            print(f"\nSniffing packets for {TIME_INTERVAL_SECONDS} seconds...")

            # Sniff packets for the specified interval
            sniffed_packets = sniff(iface=args.iface, timeout=TIME_INTERVAL_SECONDS)

            if not sniffed_packets:
                print("No packets were sniffed in the last interval.")
                # We should still append a row of zeros to keep the time-series consistent
                new_row = pd.DataFrame([[pd.Timestamp.now(), 0, 0, 0]], columns=['timestamp'] + features_to_use)
            else:
                # Process the sniffed packets
                # Note: process_packets returns a DataFrame, we expect one row for the interval
                new_row = process_packets(sniffed_packets, args.ip, f"{TIME_INTERVAL_SECONDS}S")

            if not new_row.empty:
                # Append the aggregated features to our queue
                data_queue.append(new_row[features_to_use].iloc[0].values)
                print(f"Interval data collected. Total intervals in queue: {len(data_queue)}")
            else:
                print("Warning: No data was processed for this interval.")


            # 3. Make Prediction
            if len(data_queue) == SEQUENCE_LENGTH:
                # We have enough data to make a prediction
                sequence_data = np.array(list(data_queue))

                # Scale the data
                scaled_sequence = scaler.transform(sequence_data)

                # Reshape for the model
                scaled_sequence = np.expand_dims(scaled_sequence, axis=0)

                # Predict
                predicted_rtt = model.predict(scaled_sequence)[0][0]

                # 4. Display Result
                print("\n--- Prediction Result ---")
                print(f"Predicted average RTT for the next {TIME_INTERVAL_SECONDS}s: {predicted_rtt:.4f} seconds")

                if predicted_rtt >= SEVERE_CONGESTION_THRESHOLD:
                    print("Status: \033[91mSevere Congestion Likely\033[0m") # Red
                elif predicted_rtt >= MODERATE_CONGESTION_THRESHOLD:
                    print("Status: \033[93mModerate Congestion Possible\033[0m") # Yellow
                else:
                    print("Status: \033[92mNetwork Conditions Appear Stable\033[0m") # Green
            else:
                print(f"Collecting initial data... {len(data_queue)}/{SEQUENCE_LENGTH} intervals gathered.")

    except (PermissionError, OSError):
        print(f"\nError: Permission denied to sniff on interface '{args.iface}'.")
        print("Please try running the script with administrator/root privileges (e.g., using 'sudo').")
    except KeyboardInterrupt:
        print("\nStopping real-time prediction.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
