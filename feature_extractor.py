import pandas as pd
from scapy.all import *
import numpy as np

def calculate_rtt(packets, laptop_ip):
    """
    Calculates Round-Trip Time (RTT) for TCP packets from a list of scapy packets.
    """
    sent_packets = {}
    rtt_records = []

    for pkt in packets:
        if not (TCP in pkt and IP in pkt):
            continue

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst

        if src_ip == laptop_ip and len(pkt[TCP].payload) > 0:
            sent_packets[pkt[TCP].seq] = pkt.time
        elif dst_ip == laptop_ip and pkt[TCP].flags & 0x10:
            ack = pkt[TCP].ack
            if ack in sent_packets:
                sent_time = sent_packets[ack]
                rtt = pkt.time - sent_time
                rtt_records.append({'timestamp': pkt.time, 'rtt': float(rtt)})
                del sent_packets[ack]

    return rtt_records

def resample_and_aggregate(df, interval):
    """
    Resamples the dataframe into fixed time intervals and aggregates features.
    """
    df.set_index('timestamp', inplace=True)

    agg_rules = {
        'length': 'sum',
        'rtt': 'mean',
        'is_from_laptop': 'count'
    }

    resampled_df = df.resample(interval).agg(agg_rules)
    resampled_df.rename(columns={'is_from_laptop': 'packet_count'}, inplace=True)

    resampled_df['rtt'] = resampled_df['rtt'].fillna(0)
    resampled_df.reset_index(inplace=True)

    print(f"Data resampled into {interval} intervals.")
    return resampled_df

def process_packets(packets, laptop_ip, interval):
    """
    Takes a list of scapy packets and returns a processed, aggregated DataFrame.
    This function combines the logic for use by both file-based and real-time processing.
    """
    # --- Basic Feature Extraction ---
    data = []
    for pkt in packets:
        record = {'timestamp': pkt.time, 'length': len(pkt)}
        if IP in pkt:
            record['is_from_laptop'] = 1 if pkt[IP].src == laptop_ip else 0
        data.append(record)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # --- RTT Calculation ---
    rtt_data = calculate_rtt(packets, laptop_ip)

    if rtt_data:
        rtt_df = pd.DataFrame(rtt_data)
        rtt_df['timestamp'] = pd.to_numeric(rtt_df['timestamp'], errors='coerce')
        rtt_df.dropna(subset=['timestamp'], inplace=True)
        rtt_df['timestamp'] = pd.to_datetime(rtt_df['timestamp'], unit='s')

        df = pd.merge_asof(df.sort_values('timestamp'), rtt_df.sort_values('timestamp'), on='timestamp', direction='backward')
        df['rtt'] = df['rtt'].fillna(method='ffill').fillna(0)
    else:
        df['rtt'] = 0

    # --- Resample and Aggregate Data ---
    aggregated_df = resample_and_aggregate(df, interval)

    return aggregated_df
