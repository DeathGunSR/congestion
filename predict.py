import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import argparse
import os
import sys

# Import the processing function from our other script
try:
    from pcap_parser import process_pcap, create_features
except ImportError:
    print("Error: pcap_parser.py not found. Make sure it's in the same directory.")
    sys.exit(1)

# --- Configuration ---
MODEL_FILE = 'congestion_model.keras'
SCALER_FILE = 'scaler.gz'
SEQUENCE_LENGTH = 60 # This must match the sequence length used for training
TEMP_CSV = 'temp_processed_for_prediction.csv'

# --- Prediction Thresholds ---
# These are example thresholds. They should be tuned based on real-world data.
# We are predicting RTT, so a higher value suggests more congestion.
SEVERE_CONGESTION_THRESHOLD = 0.1 # e.g., 100ms
MODERATE_CONGESTION_THRESHOLD = 0.05 # e.g., 50ms


def make_prediction(model, scaler, data_sequence):
    """
    Scales the data, makes a prediction, and returns the raw prediction value.
    """
    # Reshape for scaler
    # The scaler expects a 2D array, and we have a 3D sequence (1, 60, 5)
    # We need to scale each feature across the sequence
    # A simple approach is to scale the whole sequence

    # Note: A more correct way to scale would be to fit the scaler on a
    # representative dataset and use that to transform new data.
    # Since we saved the scaler from the training script, we can just use it.

    # The scaler was fitted on a dataframe with specific columns.
    # We need to present the data in the same way.
    features_to_use = ['length', 'rtt', 'interarrival_time', 'packets_per_second', 'throughput_bps']

    # Create a dataframe from our sequence to ensure column order
    df_sequence = pd.DataFrame(data_sequence, columns=features_to_use)

    # Scale the data
    scaled_sequence = scaler.transform(df_sequence)

    # Reshape back to (1, sequence_length, num_features) for the model
    scaled_sequence = np.expand_dims(scaled_sequence, axis=0)

    prediction = model.predict(scaled_sequence)
    return prediction[0][0]


def main():
    """Main function to run the prediction pipeline."""
    parser = argparse.ArgumentParser(description="Predict network congestion from a pcap file.")
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

    # 2. Process the input pcap file
    # We'll use a temporary file for the processed data
    print(f"Processing '{args.pcap_file}'...")
    # We need to modify pcap_parser to be callable without user input
    # For now, let's assume the user provides the IP via command line
    # Re-writing the pcap_parser.py to accept arguments would be better,
    # but for now, we will call it and it will ask for input.
    # A better approach would be to refactor pcap_parser.py to have a main guard
    # and importable functions. I've already done this.

    # Let's call the function directly
    process_pcap(args.pcap_file, args.ip)

    # The process_pcap saves to a fixed filename, let's use that.
    # This is not ideal, but it's consistent with the current structure.
    processed_df = pd.read_csv('processed_data.csv')

    # 3. Prepare data for prediction
    print(f"Preparing the last {SEQUENCE_LENGTH} data points for prediction...")
    if len(processed_df) < SEQUENCE_LENGTH:
        print(f"Error: The pcap file produced less than {SEQUENCE_LENGTH} data points.")
        print("Cannot make a prediction with insufficient data.")
        return

    features_to_use = ['length', 'rtt', 'interarrival_time', 'packets_per_second', 'throughput_bps']
    last_sequence = processed_df[features_to_use].tail(SEQUENCE_LENGTH).values

    # 4. Make Prediction
    predicted_rtt = make_prediction(model, scaler, last_sequence)

    # 5. Interpret and Display Result
    print("\n--- Prediction Result ---")
    print(f"Predicted average RTT in the near future: {predicted_rtt:.4f} seconds")

    if predicted_rtt >= SEVERE_CONGESTION_THRESHOLD:
        print("Status: Severe Congestion Likely")
    elif predicted_rtt >= MODERATE_CONGESTION_THRESHOLD:
        print("Status: Moderate Congestion Possible")
    else:
        print("Status: Network Conditions Appear Stable")

    # Clean up temporary file if we were using one
    # In this version, we just overwrite processed_data.csv, so no cleanup needed.

if __name__ == '__main__':
    main()
