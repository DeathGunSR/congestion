import pandas as pd
from scapy.all import *
from collections import defaultdict
import datetime

# --- Configuration ---
PCAP_FILE = 'traffic.pcap'
TIME_WINDOW = 1  # seconds
# Congestion periods in seconds from the start of the capture
# 2:00-4:30 -> 120s-270s
# 8:10-9:00 -> 490s-540s
CONGESTION_PERIODS = [(120, 270), (490, 540)]

def is_congested(timestamp, first_packet_time, periods):
    """Checks if a timestamp falls within any of the congestion periods."""
    if not first_packet_time:
        return 0
    relative_time = timestamp - first_packet_time
    for start, end in periods:
        if start <= relative_time < end:
            return 1
    return 0

def process_pcap(file_path, window_size):
    """
    Processes a pcap file to extract features based on time windows.
    """
    print(f"Starting to process {file_path}...")

    all_features = []
    packets_in_window = []
    window_start_time = None
    first_packet_time = None

    try:
        pcap_reader = PcapReader(file_path)
    except Scapy_Exception as e:
        print(f"Error reading pcap file: {e}")
        print("Please make sure 'traffic.pcap' exists and is a valid pcap file.")
        return None

    for packet in pcap_reader:
        if not hasattr(packet, 'time'):
            continue

        packet_time = float(packet.time)

        if first_packet_time is None:
            first_packet_time = packet_time
            window_start_time = packet_time

        # If the packet is outside the current window, process the previous window
        if packet_time >= window_start_time + window_size:
            if packets_in_window:
                # Feature Extraction
                total_bytes = sum(len(p) for p in packets_in_window)
                packet_count = len(packets_in_window)

                # Device count (unique source MAC addresses)
                # We check for Dot11 layer and addr2 (source address)
                devices = {p.addr2 for p in packets_in_window if hasattr(p, 'addr2') and p.addr2}
                device_count = len(devices)

                # Frame counts (Management, Control, Data)
                mgmt_frames = 0
                ctrl_frames = 0
                data_frames = 0
                for p in packets_in_window:
                    if p.haslayer(Dot11):
                        # Type 0: Management, 1: Control, 2: Data
                        if p.type == 0:
                            mgmt_frames += 1
                        elif p.type == 1:
                            ctrl_frames += 1
                        elif p.type == 2:
                            data_frames += 1

                # Labeling
                label = is_congested(window_start_time, first_packet_time, CONGESTION_PERIODS)

                all_features.append({
                    'timestamp': datetime.datetime.fromtimestamp(window_start_time).strftime('%Y-%m-%d %H:%M:%S'),
                    'packet_count': packet_count,
                    'total_bytes': total_bytes,
                    'device_count': device_count,
                    'mgmt_frame_count': mgmt_frames,
                    'ctrl_frame_count': ctrl_frames,
                    'data_frame_count': data_frames,
                    'is_congested': label
                })

            # Start a new window
            packets_in_window = []
            window_start_time += window_size
            # Ensure window_start_time catches up if there are gaps in traffic
            while window_start_time + window_size <= packet_time:
                 # Create empty time windows for gaps
                label = is_congested(window_start_time, first_packet_time, CONGESTION_PERIODS)
                all_features.append({
                    'timestamp': datetime.datetime.fromtimestamp(window_start_time).strftime('%Y-%m-%d %H:%M:%S'),
                    'packet_count': 0, 'total_bytes': 0, 'device_count': 0,
                    'mgmt_frame_count': 0, 'ctrl_frame_count': 0, 'data_frame_count': 0,
                    'is_congested': label
                })
                window_start_time += window_size


        packets_in_window.append(packet)

    # Process the last window
    if packets_in_window:
        total_bytes = sum(len(p) for p in packets_in_window)
        packet_count = len(packets_in_window)
        devices = {p.addr2 for p in packets_in_window if hasattr(p, 'addr2') and p.addr2}
        device_count = len(devices)
        mgmt_frames = sum(1 for p in packets_in_window if p.haslayer(Dot11) and p.type == 0)
        ctrl_frames = sum(1 for p in packets_in_window if p.haslayer(Dot11) and p.type == 1)
        data_frames = sum(1 for p in packets_in_window if p.haslayer(Dot11) and p.type == 2)
        label = is_congested(window_start_time, first_packet_time, CONGESTION_PERIODS)
        all_features.append({
            'timestamp': datetime.datetime.fromtimestamp(window_start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'packet_count': packet_count, 'total_bytes': total_bytes, 'device_count': device_count,
            'mgmt_frame_count': mgmt_frames, 'ctrl_frame_count': ctrl_frames, 'data_frame_count': data_frames,
            'is_congested': label
        })

    if not all_features:
        print("No packets were processed. Is the pcap file empty or invalid?")
        return None

    return pd.DataFrame(all_features)

if __name__ == "__main__":
    features_df = process_pcap(PCAP_FILE, TIME_WINDOW)
    if features_df is not None:
        output_file = 'features.csv'
        features_df.to_csv(output_file, index=False)
        print(f"Feature extraction complete. Data saved to {output_file}")
        print(f"Total time windows processed: {len(features_df)}")
        print("Label distribution:")
        print(features_df['is_congested'].value_counts())
