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

def get_network_interface():
    try:
        from scapy.arch import get_windows_if_list
        return [iface['name'] for iface in get_windows_if_list()]
    except ImportError:
        return os.listdir('/sys/class/net/')

def main():
    parser = argparse.ArgumentParser(description="Predict network congestion trends in real-time.")
    parser.add_argument("--ip", required=True, help="The IP address of this machine.")
    parser.add_argument("--iface", required=True, help="The network interface to sniff on.")
    parser.add_argument("--activity", required=True, help="The type of activity: e.g., 'web'.")
    args = parser.parse_args()

    print("Loading resources...")
    test_accuracy = 0.0
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        with open(ACTIVITY_MAP_FILE, 'r') as f:
            activity_map = json.load(f)
        with open('model_stats.json', 'r') as f:
            stats = json.load(f)
        test_accuracy = stats.get('test_accuracy', 0.0)
    except FileNotFoundError:
        print("Warning: model_stats.json not found. Accuracy will be displayed as 0%.")
    except Exception as e:
        print(f"Error loading files: {e}\nPlease run pcap_parser.py and train_model.py first.")
        return

    if args.activity not in activity_map:
        print(f"Error: Activity '{args.activity}' not found. Available: {list(activity_map.keys())}")
        return

    activity_index = activity_map[args.activity]
    print(f"Using activity: '{args.activity}' (index: {activity_index})")

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_points_to_keep = int(PLOT_HISTORY_SECONDS / TIME_INTERVAL_SECONDS)
    time_steps = deque(maxlen=plot_points_to_keep)
    actual_rtts = deque(maxlen=plot_points_to_keep)
    prediction_history = deque(maxlen=plot_points_to_keep)

    features_to_use = [
        'length_sum', 'rtt_mean', 'rtt_min', 'rtt_max', 'rtt_std',
        'packet_count', 'lost_packet_count', 'rtt_mean_trend',
        'packet_count_trend', 'length_sum_trend', 'rtt_volatility'
    ]
    data_queue = deque(maxlen=SEQUENCE_LENGTH)

    print(f"\nStarting real-time congestion prediction on interface '{args.iface}'...")
    print("Press Ctrl+C to stop.")

    try:
        time_counter = 0
        while True:
            print(f"\n[{pd.Timestamp.now()}] Sniffing for {TIME_INTERVAL_SECONDS} seconds...")
            sniffed_packets = sniff(iface=args.iface, timeout=TIME_INTERVAL_SECONDS)

            new_row_df = process_packets(sniffed_packets, args.ip, f"{int(TIME_INTERVAL_SECONDS * 1000)}ms")

            if not new_row_df.empty:
                actual_rtt_value = new_row_df['rtt_mean'].iloc[0]
                data_queue.append(new_row_df[features_to_use].iloc[0].values)
                print(f"Interval data collected. Actual RTT: {actual_rtt_value:.4f}s.")
            else:
                actual_rtt_value = 0.0
                data_queue.append(np.zeros(len(features_to_use)))
                print("No packets sniffed. Appending zeros.")

            time_steps.append(time_counter)
            actual_rtts.append(actual_rtt_value)

            # --- Prediction Logic ---
            color = 'gray' # Default color if no prediction is made
            if len(data_queue) == SEQUENCE_LENGTH:
                ts_data = np.array(list(data_queue))
                ts_df = pd.DataFrame(ts_data, columns=features_to_use)
                scaled_ts_data = scaler.transform(ts_df)

                model_input = [
                    np.expand_dims(scaled_ts_data, axis=0),
                    np.array([activity_index])
                ]
                prediction_prob = model.predict(model_input, verbose=0)[0][0]
                status = "Worsening" if prediction_prob > 0.5 else "Stable"
                color = '#d62728' if status == "Worsening" else '#2ca02c' # Red / Green
                print(f"Model Prediction: {status} (Confidence: {prediction_prob:.2f})")

            prediction_history.append(color)

            # --- Plotting Logic ---
            ax.clear()
            ax.plot(list(time_steps), list(actual_rtts), 'o-', color='#1f77b4', label='Actual RTT')

            # Shade the background based on historical predictions
            for i in range(1, len(time_steps)):
                ax.axvspan(time_steps[i-1], time_steps[i], facecolor=prediction_history[i], alpha=0.3)

            # Add dummy patches for the legend
            legend_patches = [
                plt.Rectangle((0,0),1,1, color='#2ca02c', alpha=0.3),
                plt.Rectangle((0,0),1,1, color='#d62728', alpha=0.3)
            ]

            # Get the original legend handles and labels
            handles, labels = ax.get_legend_handles_labels()

            # Append new patches and labels
            ax.legend(handles + legend_patches, labels + ['Predicted: Stable', 'Predicted: Worsening'], loc='upper left')

            ax.set_xlabel(f"Time Intervals ({TIME_INTERVAL_SECONDS}s each)")
            ax.set_ylabel("Round-Trip Time (s)")
            ax.set_title(
                f"Real-Time RTT & Congestion Prediction (Activity: {args.activity})\n"
                f"Model Test Accuracy: {test_accuracy*100:.2f}%"
            )
            ax.legend()
            ax.grid(True)
            fig.canvas.draw()
            plt.pause(0.01)

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
