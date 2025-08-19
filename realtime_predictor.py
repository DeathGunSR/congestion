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
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_FILE = 'congestion_model.keras'
SCALER_FILE = 'scaler.gz'
SEQUENCE_LENGTH = 10
TIME_INTERVAL_SECONDS = 0.5

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
        return os.listdir('/sys/class/net/')

def main():
    """Main function to run the real-time prediction pipeline."""
    parser = argparse.ArgumentParser(description="Predict network congestion in real-time with plotting.")
    parser.add_argument("--ip", required=True, help="The IP address of this machine.")
    parser.add_argument("--iface", required=False, help="The network interface to sniff on.")
    args = parser.parse_args()

    if not args.iface:
        print("No network interface specified. Available interfaces:")
        for iface in get_network_interface():
            print(f" - {iface}")
        print("\nPlease run the script again with the --iface argument.")
        return

    print("Loading trained model and scaler...")
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    except (IOError, ImportError) as e:
        print(f"Error loading model or scaler: {e}\nPlease run train_model.py first.")
        return

    # --- Plotting Setup ---
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 6))
    time_steps = []
    actual_rtts = []
    predicted_rtts = []

    # --- Data & Prediction Setup ---
    data_queue = deque(maxlen=SEQUENCE_LENGTH)
    features_to_use = ['length', 'rtt', 'packet_count']
    last_prediction = None

    print(f"\nStarting real-time congestion prediction on interface '{args.iface}'...")
    print("Press Ctrl+C to stop. The plot window may become unresponsive but will update.")

    try:
        time_counter = 0
        while True:
            print(f"\n[{pd.Timestamp.now()}] Sniffing for {TIME_INTERVAL_SECONDS} seconds...")
            sniffed_packets = sniff(iface=args.iface, timeout=TIME_INTERVAL_SECONDS)

            new_row_df = process_packets(sniffed_packets, args.ip, f"{TIME_INTERVAL_SECONDS}S")

            if not new_row_df.empty:
                actual_rtt_value = new_row_df['rtt'].iloc[0]
                data_queue.append(new_row_df[features_to_use].iloc[0].values)
                print(f"Interval data collected. Actual RTT: {actual_rtt_value:.4f}s. Queue size: {len(data_queue)}")
            else:
                actual_rtt_value = 0.0 # Assume 0 RTT if no packets
                zero_data = np.zeros(len(features_to_use))
                data_queue.append(zero_data)
                print(f"No packets sniffed. Appending zeros. Queue size: {len(data_queue)}")

            # --- Prediction Logic ---
            if len(data_queue) == SEQUENCE_LENGTH:
                # Plot the actual RTT from this interval and the prediction made in the *last* interval
                time_steps.append(time_counter)
                actual_rtts.append(actual_rtt_value)
                predicted_rtts.append(last_prediction if last_prediction is not None else 0) # Append last prediction

                # --- Plotting Update ---
                ax.clear()
                ax.plot(time_steps, actual_rtts, 'bo-', label='Actual RTT')
                ax.plot(time_steps, predicted_rtts, 'ro--', label='Predicted RTT')
                ax.set_xlabel(f"Time Intervals ({TIME_INTERVAL_SECONDS}s each)")
                ax.set_ylabel("Round-Trip Time (s)")
                ax.set_title("Real-Time vs. Predicted RTT")
                ax.legend()
                ax.grid(True)
                fig.canvas.draw()
                plt.pause(0.01)

                # --- Make new prediction for the NEXT interval ---
                sequence_data = np.array(list(data_queue))
                scaled_sequence = scaler.transform(sequence_data)
                scaled_sequence = np.expand_dims(scaled_sequence, axis=0)

                # This prediction is for the *next* time step
                last_prediction = model.predict(scaled_sequence, verbose=0)[0][0]
                print(f"Prediction for next interval: {last_prediction:.4f}s")

            else:
                print(f"Collecting initial data... {len(data_queue)}/{SEQUENCE_LENGTH} intervals gathered.")

            time_counter += 1

    except (PermissionError, OSError):
        print(f"\nError: Permission denied to sniff on interface '{args.iface}'. Try with sudo/admin.")
    except KeyboardInterrupt:
        print("\nStopping real-time prediction.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        plt.ioff()
        plt.show() # Keep the final plot open

if __name__ == '__main__':
    main()
