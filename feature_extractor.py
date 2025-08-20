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

def resample_and_engineer_features(df, interval):
    """
    Resamples the dataframe and engineers advanced features for trend and volatility.
    """
    df.set_index('timestamp', inplace=True)

    # Define aggregation rules for resampling
    agg_rules = {
        'length': 'sum',
        'rtt': ['mean', 'min', 'max', 'std'], # Calculate multiple stats for RTT
        'is_from_laptop': 'count'
    }

    resampled_df = df.resample(interval).agg(agg_rules)

    # Flatten the multi-level column index from aggregation
    resampled_df.columns = ['_'.join(col).strip() for col in resampled_df.columns.values]
    resampled_df.rename(columns={'is_from_laptop_count': 'packet_count'}, inplace=True)

    # --- Engineer New Features ---

    # Fill NaNs that result from empty intervals before calculating trends
    resampled_df.fillna(0, inplace=True)

    # 1. Momentum/Trend Features (the change from the previous interval)
    resampled_df['rtt_mean_trend'] = resampled_df['rtt_mean'].diff().fillna(0)
    resampled_df['packet_count_trend'] = resampled_df['packet_count'].diff().fillna(0)
    resampled_df['length_sum_trend'] = resampled_df['length_sum'].diff().fillna(0)

    # 2. Volatility Feature
    # A simple measure of volatility: the difference between max and min RTT
    resampled_df['rtt_volatility'] = resampled_df['rtt_max'] - resampled_df['rtt_min']

    resampled_df.reset_index(inplace=True)

    print(f"Data resampled and advanced features engineered for {interval} intervals.")
    return resampled_df

def process_packets(packets, laptop_ip, interval):
    """
    Takes a list of scapy packets and returns a processed, feature-engineered DataFrame.
    """
    data = []
    for pkt in packets:
        record = {'timestamp': pkt.time, 'length': len(pkt)}
        if IP in pkt:
            record['is_from_laptop'] = 1 if pkt[IP].src == laptop_ip else 0
        data.append(record)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').dropna()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    rtt_data = calculate_rtt(packets, laptop_ip)

    if rtt_data:
        rtt_df = pd.DataFrame(rtt_data)
        rtt_df['timestamp'] = pd.to_numeric(rtt_df['timestamp'], errors='coerce').dropna()
        rtt_df['timestamp'] = pd.to_datetime(rtt_df['timestamp'], unit='s')
        df = pd.merge_asof(df.sort_values('timestamp'), rtt_df.sort_values('timestamp'), on='timestamp', direction='backward')
        df['rtt'] = df['rtt'].fillna(method='ffill').fillna(0)
    else:
        df['rtt'] = 0

    # Use the new feature engineering function
    engineered_df = resample_and_engineer_features(df, interval)

    return engineered_df
