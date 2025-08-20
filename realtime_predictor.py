import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import argparse
import os
import sys
import json
from collections import deque
from scapy.all import sniff
from feature_extractor import process_packets
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_FILE = 'congestion_model.keras'
SCALER_FILE = 'scaler.gz'
ACTIVITY_MAP_FILE = 'activity_map.json'
SEQUENCE_LENGTH = 10
TIME_INTERVAL_SECONDS = 0.5

# --- Prediction Thresholds ---
SEVERE_CONGESTION_THRESHOLD = 0.1
MODERATE_CONGESTION_THRESHOLD = 0.05

def get_network_interface():
    try:
        from scapy.arch import get_windows_if_list
        return [iface['name'] for iface in get_windows_if_list()]
    except ImportError:
        return os.listdir('/sys/class/net/')

def main():
    parser = argparse.ArgumentParser(description="Predict network congestion in real-time with plotting.")
    parser.add_argument("--ip", required=True, help="The IP address of this machine.")
    parser.add_argument("--iface", required=True, help="The network interface to sniff on.")
    parser.add_argument("--activity", required=True, help="The type of activity being performed (e.g., 'web_browsing').")
    args = parser.parse_args()

    print("Loading trained model, scaler, and activity map...")
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        with open(ACTIVITY_MAP_FILE, 'r') as f:
            activity_map = json.load(f)
    except Exception as e:
        print(f"Error loading files: {e}\nPlease run pcap_parser.py and train_model.py first.")
        return

    if args.activity not in activity_map:
        print(f"Error: Activity '{args.activity}' not found in activity_map.json.")
        print(f"Available activities are: {list(activity_map.keys())}")
        return

    activity_index = activity_map[args.activity]
    print(f"Using activity: '{args.activity}' (index: {activity_index})")

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    time_steps, actual_rtts, predicted_rtts = [], [], []

    features_to_use = [
        'length_sum', 'rtt_mean', 'rtt_min', 'rtt_max', 'rtt_std',
        'packet_count', 'rtt_mean_trend', 'packet_count_trend',
        'length_sum_trend', 'rtt_volatility'
    ]
    data_queue = deque(maxlen=SEQUENCE_LENGTH)
    last_prediction = None

    print(f"\nStarting real-time congestion prediction on interface '{args.iface}'...")
    print("Press Ctrl+C to stop.")

    try:
        time_counter = 0
        while True:
            print(f"\n[{pd.Timestamp.now()}] Sniffing for {TIME_INTERVAL_SECONDS} seconds...")
            sniffed_packets = sniff(iface=args.iface, timeout=TIME_INTERVAL_SECONDS)
            new_row_df = process_packets(sniffed_packets, args.ip, f"{TIME_INTERVAL_SECONDS}S")

            if not new_row_df.empty:
                actual_rtt_value = new_row_df['rtt_mean'].iloc[0]
                data_queue.append(new_row_df[features_to_use].iloc[0].values)
            else:
                actual_rtt_value = 0.0
                data_queue.append(np.zeros(len(features_to_use)))

            print(f"Interval data collected. Actual RTT: {actual_rtt_value:.4f}s. Queue size: {len(data_queue)}")

            if len(data_queue) == SEQUENCE_LENGTH:
                time_steps.append(time_counter)
                actual_rtts.append(actual_rtt_value)
                predicted_rtts.append(last_prediction if last_prediction is not None else 0)

                ax.clear()
                ax.plot(time_steps, actual_rtts, 'bo-', label='Actual RTT')
                ax.plot(time_steps, predicted_rtts, 'ro--', label='Predicted RTT')
                ax.set_xlabel(f"Time Intervals ({TIME_INTERVAL_SECONDS}s each)")
                ax.set_ylabel("Round-Trip Time (s)")
                ax.set_title(f"Real-Time vs. Predicted RTT (Activity: {args.activity})")
                ax.legend()
                ax.grid(True)
                fig.canvas.draw()
                plt.pause(0.01)

                ts_data = np.array(list(data_queue))
                scaled_ts_data = scaler.transform(ts_data)

                # Prepare inputs for the multi-input model
                model_input = [
                    np.expand_dims(scaled_ts_data, axis=0), # Time-series input
                    np.array([activity_index]) # Categorical input
                ]

                last_prediction = model.predict(model_input, verbose=0)[0][0]
                print(f"Prediction for next interval: {last_prediction:.4f}s")
            else:
                print(f"Collecting initial data... {len(data_queue)}/{SEQUENCE_LENGTH} intervals gathered.")

            time_counter += 1

    except (PermissionError, OSError):
        print(f"\nError: Permission denied. Try with sudo/admin.")
    except KeyboardInterrupt:
        print("\nStopping prediction.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()
