import pandas as pd
from scapy.all import *
import numpy as np
import time

# --- Configuration ---
PCAP_FILE = 'traffic.pcap'
PROCESSED_CSV_FILE = 'processed_data.csv'
LAPTOP_IP = '192.168.1.100' # Please adjust if necessary

def calculate_rtt(packets, laptop_ip):
    """
    Calculates Round-Trip Time (RTT) for TCP packets.
    This function has been improved to better match TCP SEQ/ACK pairs.
    """
    sent_packets = {}
    rtt_records = []

    for pkt in packets:
        if not (TCP in pkt and IP in pkt):
            continue

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst

        # Packet sent from the laptop (DATA)
        if src_ip == laptop_ip and len(pkt[TCP].payload) > 0:
            # Store the time the packet was sent, keyed by SEQ number
            sent_packets[pkt[TCP].seq] = pkt.time

        # Packet received by the laptop (ACK)
        elif dst_ip == laptop_ip and pkt[TCP].flags & 0x10: # Check for ACK flag
            # Find the corresponding sent packet
            ack = pkt[TCP].ack
            if ack in sent_packets:
                sent_time = sent_packets[ack]
                rtt = pkt.time - sent_time
                rtt_records.append({
                    'timestamp': pkt.time,
                    'rtt': float(rtt)
                })
                # Remove matched SEQ to prevent duplicates and keep dict small
                del sent_packets[ack]

    return rtt_records


def create_features(df):
    """
    Engineers features from the raw packet data.
    """
    # Ensure dataframe is sorted by time
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Calculate inter-arrival time
    df['interarrival_time'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

    # Calculate packets per second (rolling window of 1 second)
    df.set_index('timestamp', inplace=True)
    df['packets_per_second'] = df['length'].rolling('1s').count()

    # Calculate throughput in bytes per second (rolling window of 1 second)
    df['throughput_bps'] = df['length'].rolling('1s').sum()

    df.reset_index(inplace=True)
    df.fillna(0, inplace=True)

    return df


def process_pcap(pcap_file, laptop_ip):
    """
    Reads a pcap file, extracts features, engineers new features,
    and saves the data to a CSV file.
    """
    print(f"Reading pcap file: {pcap_file}...")
    try:
        packets = rdpcap(pcap_file)
    except FileNotFoundError:
        print(f"Error: The file '{pcap_file}' was not found.")
        print("Please make sure the pcap file is in the same directory.")
        return

    print(f"Total packets read: {len(packets)}")

    # --- Basic Feature Extraction ---
    data = []
    for pkt in packets:
        record = {'timestamp': pkt.time, 'length': len(pkt)}
        if IP in pkt:
            record['is_from_laptop'] = 1 if pkt[IP].src == laptop_ip else 0
        data.append(record)

    if not data:
        print("No packets were processed. Exiting.")
        return

    df = pd.DataFrame(data)
    # Coerce errors will turn problematic values into NaT (Not a Time)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp was invalid
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    print("Basic feature extraction complete.")

    # --- RTT Calculation ---
    print("Calculating RTT for TCP packets...")
    rtt_data = calculate_rtt(packets, laptop_ip)

    if rtt_data:
        rtt_df = pd.DataFrame(rtt_data)
        rtt_df['timestamp'] = pd.to_datetime(rtt_df['timestamp'], unit='s')

        # Merge RTT data with the main dataframe
        df = pd.merge_asof(df.sort_values('timestamp'), rtt_df.sort_values('timestamp'), on='timestamp', direction='backward')
        df['rtt'] = df['rtt'].fillna(method='ffill').fillna(0) # Forward fill and then zero fill
        print(f"RTT calculation complete. Found {len(rtt_df)} RTT values.")
    else:
        print("Could not calculate any RTT values. RTT column will be zero.")
        df['rtt'] = 0

    # --- Advanced Feature Engineering ---
    print("Engineering additional features (inter-arrival time, throughput)...")
    df = create_features(df)

    # --- Save to CSV ---
    # Select final columns for the model
    final_columns = [
        'timestamp',
        'length',
        'is_from_laptop',
        'rtt',
        'interarrival_time',
        'packets_per_second',
        'throughput_bps'
    ]
    df = df[final_columns]

    df.to_csv(PROCESSED_CSV_FILE, index=False)
    print(f"Processed data with engineered features saved to {PROCESSED_CSV_FILE}")


if __name__ == '__main__':
    # You might need to change this IP depending on your network setup
    # It should be the IP of the machine where the capture was taken
    laptop_ip_address = input(f"Enter the laptop's IP address (default: {LAPTOP_IP}): ")
    if not laptop_ip_address:
        laptop_ip_address = LAPTOP_IP

    process_pcap(PCAP_FILE, laptop_ip_address)
    print("\nScript finished.")
