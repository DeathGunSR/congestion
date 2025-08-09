import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import argparse
import os
import sys

# Import the processing function from our other script
try:
    # We now only need the main processing function
    from pcap_parser import process_pcap
except ImportError:
    print("Error: pcap_parser.py not found. Make sure it's in the same directory.")
    sys.exit(1)

# --- Configuration ---
MODEL_FILE = 'congestion_model.keras'
SCALER_FILE = 'scaler.gz'
# This must match the sequence length used for training (12 intervals of 5s)
SEQUENCE_LENGTH = 12
PROCESSED_CSV_FILE = 'processed_data.csv' # The parser saves to this file

# --- Prediction Thresholds ---
# These are example thresholds for average RTT in a 5s interval
SEVERE_CONGESTION_THRESHOLD = 0.1
MODERATE_CONGESTION_THRESHOLD = 0.05

def make_prediction(model, scaler, data_sequence):
    """
    Scales the data, makes a prediction, and returns the raw prediction value.
    """
    # The scaler was fitted on a dataframe with specific columns.
    # We need to present the data in the same way.
    features_to_use = ['length', 'rtt', 'packet_count']

    df_sequence = pd.DataFrame(data_sequence, columns=features_to_use)

    # Scale the data
    scaled_sequence = scaler.transform(df_sequence)

    # Reshape to (1, sequence_length, num_features) for the model
    scaled_sequence = np.expand_dims(scaled_sequence, axis=0)

    prediction = model.predict(scaled_sequence)
    return prediction[0][0]

def main():
    """Main function to run the prediction pipeline."""
    parser = argparse.ArgumentParser(description="Predict network congestion from a pcap file based on 5-second intervals.")
    parser.add_argument("pcap_file", help="Path to the .pcap file for analysis.")
    parser.add_argument("--ip", help="The IP address of the local machine (laptop).", default="192.168.1.100")
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
        print(f"Error loading model or scaler: {e}")
        print("Please run train_model.py first to generate these files.")
        return

    # 2. Process the input pcap file using the updated parser
    print(f"Processing '{args.pcap_file}' into 5-second intervals...")
    process_pcap(args.pcap_file, args.ip)

    # Load the aggregated data
    processed_df = pd.read_csv(PROCESSED_CSV_FILE)

    # 3. Prepare data for prediction
    print(f"Preparing the last {SEQUENCE_LENGTH} time intervals for prediction...")
    if len(processed_df) < SEQUENCE_LENGTH:
        print(f"Error: The pcap file produced less than {SEQUENCE_LENGTH} time intervals.")
        print("Cannot make a prediction with insufficient data.")
        return

    features_to_use = ['length', 'rtt', 'packet_count']
    last_sequence = processed_df[features_to_use].tail(SEQUENCE_LENGTH).values

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
