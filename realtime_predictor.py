import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import argparse
import os
import sys
import json
from collections import deque
from scapy.all import sniff, TCP
from feature_extractor import process_packets
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_FILE = 'congestion_model.keras'
SCALER_FILE = 'scaler.gz'
ACTIVITY_MAP_FILE = 'activity_map.json'
SEQUENCE_LENGTH = 10
TIME_INTERVAL_SECONDS = 0.5
PLOT_HISTORY_SECONDS = 45

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
    parser.add_argument("--bias-correction", type=float, default=0.0, help="A constant to subtract from predictions for manual calibration.")
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

    # Initialize data stores
    plot_points_to_keep = int(PLOT_HISTORY_SECONDS / TIME_INTERVAL_SECONDS)
    time_steps = deque(maxlen=plot_points_to_keep)
    actual_rtts = deque(maxlen=plot_points_to_keep)
    predicted_rtts = deque(maxlen=plot_points_to_keep)
    error_history = deque(maxlen=plot_points_to_keep) # For calculating rolling MAPE

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

            if sniffed_packets:
                new_row_df = process_packets(sniffed_packets, args.ip, f"{int(TIME_INTERVAL_SECONDS * 1000)}ms")
                actual_rtt_value = new_row_df['rtt_mean'].iloc[0]
                data_queue.append(new_row_df[features_to_use].iloc[0].values)
                print(f"Interval data collected. Actual RTT: {actual_rtt_value:.4f}s. Queue size: {len(data_queue)}")
            else:
                print("No packets sniffed in the last interval.")
                actual_rtt_value = 0.0
                data_queue.append(np.zeros(len(features_to_use)))
                print(f"Appending zeros. Queue size: {len(data_queue)}")

            if len(data_queue) == SEQUENCE_LENGTH:
                # Check for TCP packets in the last interval before predicting
                has_tcp = any(TCP in pkt for pkt in sniffed_packets)

                if has_tcp:
                    # --- Make new prediction for the NEXT interval ---
                    ts_data = np.array(list(data_queue))

                    # Convert to DataFrame to preserve feature names for the scaler
                    ts_df = pd.DataFrame(ts_data, columns=features_to_use)
                    scaled_ts_data = scaler.transform(ts_df)

                    model_input = [
                        np.expand_dims(scaled_ts_data, axis=0),
                        np.array([activity_index])
                    ]

                    raw_prediction = model.predict(model_input, verbose=0)[0][0]
                    # Apply bias correction and clip at zero
                    current_prediction = max(0.0, raw_prediction - args.bias_correction)
                else:
                    # If no TCP packets, override prediction to 0
                    print("No TCP packets detected. Overriding prediction to 0.")
                    current_prediction = 0.0

                # Use the prediction from the previous step for this interval's error calculation
                pred_for_error_plot = last_prediction if last_prediction is not None else 0

                # Calculate MAPE only if actual RTT is not zero
                if actual_rtt_value > 0:
                    error = np.abs((actual_rtt_value - pred_for_error_plot) / actual_rtt_value) * 100
                    error_history.append(error)

                mape = np.mean(error_history) if error_history else 0.0

                # Update data for plotting
                time_steps.append(time_counter)
                actual_rtts.append(actual_rtt_value)
                predicted_rtts.append(pred_for_error_plot)

                # Update the plot
                ax.clear()
                ax.plot(list(time_steps), list(actual_rtts), 'bo-', label='Actual RTT')
                ax.plot(list(time_steps), list(predicted_rtts), 'ro--', label='Predicted RTT')
                ax.set_xlabel(f"Time Intervals ({TIME_INTERVAL_SECONDS}s each)")
                ax.set_ylabel("Round-Trip Time (s)")
                ax.set_title(f"Real-Time vs. Predicted RTT (MAPE: {mape:.2f}%)")
                ax.legend()
                ax.grid(True)
                fig.canvas.draw()
                plt.pause(0.01)

                # Store the prediction for the next iteration's plot
                last_prediction = current_prediction
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
