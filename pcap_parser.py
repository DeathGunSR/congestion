import os
import json
import pandas as pd
import argparse
from scapy.all import rdpcap
from feature_extractor import process_packets

# --- Configuration ---
DATA_DIR = 'data'
PROCESSED_CSV_FILE = 'processed_data.csv'
ACTIVITY_MAP_FILE = 'activity_map.json'
TIME_INTERVAL = '500ms'

def get_activity_from_filename(filename):
    """Extracts the base activity name from the pcap filename."""
    return os.path.basename(filename).split('_')[0].split('.')[0]

def main():
    """
    Parses all pcap files in the DATA_DIR, processes them, adds an activity
    label, and saves the combined data to a single CSV file.
    """
    parser = argparse.ArgumentParser(
        description="Parse .pcap files, extract features, and create a unified dataset."
    )
    parser.add_argument("--ip", required=True, help="The IP address of the machine where the .pcap files were captured.")
    args = parser.parse_args()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory '{DATA_DIR}'. Please add your .pcap files there.")
        return

    pcap_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pcap')]

    if not pcap_files:
        print(f"No .pcap files found in the '{DATA_DIR}' directory.")
        return

    all_dfs = []
    activity_labels = sorted(list(set([get_activity_from_filename(f) for f in pcap_files])))
    activity_map = {label: i for i, label in enumerate(activity_labels)}

    print("Found the following activities:")
    for label, index in activity_map.items():
        print(f"  - {label}: {index}")

    # Save the activity mapping for the prediction script
    with open(ACTIVITY_MAP_FILE, 'w') as f:
        json.dump(activity_map, f, indent=4)
    print(f"Activity map saved to '{ACTIVITY_MAP_FILE}'")

    print(f"Using IP address {args.ip} for all files.")

    for pcap_file in pcap_files:
        filepath = os.path.join(DATA_DIR, pcap_file)
        print(f"\nProcessing file: {filepath}...")

        try:
            packets = rdpcap(filepath)
            print(f"Total packets read: {len(packets)}")
        except Exception as e:
            print(f"Error reading pcap file '{filepath}': {e}")
            continue

        aggregated_df = process_packets(packets, args.ip, TIME_INTERVAL)

        if not aggregated_df.empty:
            activity_name = get_activity_from_filename(pcap_file)
            aggregated_df['activity_type'] = activity_map[activity_name]
            all_dfs.append(aggregated_df)
        else:
            print(f"No data processed for {pcap_file}.")

    if not all_dfs:
        print("No data was processed from any pcap file.")
        return

    # Combine all dataframes and save
    final_df = pd.concat(all_dfs, ignore_index=True).sort_values(by='timestamp').reset_index(drop=True)
    final_df.to_csv(PROCESSED_CSV_FILE, index=False)
    print(f"\nCombined and processed data from {len(pcap_files)} files saved to '{PROCESSED_CSV_FILE}'")

if __name__ == '__main__':
    main()
    print("\nScript finished.")
