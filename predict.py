import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import argparse
import os
import sys
from scapy.all import rdpcap
from feature_extractor import process_packets

# --- Configuration ---
MODEL_FILE = 'congestion_model.keras'
SCALER_FILE = 'scaler.gz'
SEQUENCE_LENGTH = 2
TIME_INTERVAL = '0.5S'

# --- Prediction Thresholds ---
SEVERE_CONGESTION_THRESHOLD = 0.1
MODERATE_CONGESTION_THRESHOLD = 0.05

def make_prediction(model, scaler, data_sequence):
    """
    Scales the data, makes a prediction, and returns the raw prediction value.
    """
    features_to_use = ['length', 'rtt', 'packet_count']
    df_sequence = pd.DataFrame(data_sequence, columns=features_to_use)
    scaled_sequence = scaler.transform(df_sequence)
    scaled_sequence = np.expand_dims(scaled_sequence, axis=0)
    prediction = model.predict(scaled_sequence)
    return prediction[0][0]

def main():
    """Main function to run the prediction pipeline."""
    parser = argparse.ArgumentParser(description="Predict network congestion from a pcap file.")
    parser.add_argument("pcap_file", help="Path to the .pcap file for analysis.")
    parser.add_argument("--ip", required=True, help="The IP address of the local machine (laptop).")
    args = parser.parse_args()

    if not os.path.exists(args.pcap_file):
        print(f"Error: Pcap file not found at '{args.pcap_file}'")
        return

    # 1. Load Model and Scaler
    print("Loading trained model and scaler...")
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    except (IOError, ImportError) as e:
        print(f"Error loading model or scaler: {e}\nPlease run train_model.py first.")
        return

    # 2. Process the input pcap file
    print(f"Processing '{args.pcap_file}'...")
    try:
        packets = rdpcap(args.pcap_file)
    except Exception as e:
        print(f"Error reading pcap file: {e}")
        return

    aggregated_df = process_packets(packets, args.ip, TIME_INTERVAL)

    # 3. Prepare data for prediction
    print(f"Preparing the last {SEQUENCE_LENGTH} time intervals for prediction...")
    if len(aggregated_df) < SEQUENCE_LENGTH:
        print(f"Error: The pcap file produced less than {SEQUENCE_LENGTH} time intervals.")
        print("Cannot make a prediction with insufficient data.")
        return

    features_to_use = ['length', 'rtt', 'packet_count']
    last_sequence = aggregated_df[features_to_use].tail(SEQUENCE_LENGTH).values

    # 4. Make Prediction
    predicted_rtt = make_prediction(model, scaler, last_sequence)

    # 5. Interpret and Display Result
    print("\n--- Prediction Result ---")
    print(f"Predicted average RTT for the next 5-second interval: {predicted_rtt:.4f} seconds")

    if predicted_rtt >= SEVERE_CONGESTION_THRESHOLD:
        print("Status: Severe Congestion Likely")
    elif predicted_rtt >= MODERATE_CONGESTION_THRESHOLD:
        print("Status: Moderate Congestion Possible")
    else:
        print("Status: Network Conditions Appear Stable")

if __name__ == '__main__':
    main()
