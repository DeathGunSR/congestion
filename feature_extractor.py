import pandas as pd
from scapy.all import *
import numpy as np

PACKET_LOSS_TIMEOUT = 3.0 # Seconds, increased to be more conservative

def analyze_tcp_flow(packets, laptop_ip):
    """
    Analyzes TCP packets to calculate RTT and detect lost packets.
    Returns a list of RTT records and a list of timestamps for lost packets.
    """
    sent_packets = {}
    rtt_records = []
    lost_packets_ts = []

    if not packets:
        return rtt_records, lost_packets_ts

    # Get the time of the last packet to establish a 'now' for timeout checks
    last_packet_time = packets[-1].time

    # First pass: Match ACKs to calculate RTT
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
                sent_time = sent_packets.pop(ack) # Remove as it's acknowledged
                rtt = pkt.time - sent_time
                rtt_records.append({'timestamp': pkt.time, 'rtt': float(rtt)})

    # Second pass: Check remaining unacknowledged packets for timeouts
    for seq, timestamp in list(sent_packets.items()):
        if last_packet_time - timestamp > PACKET_LOSS_TIMEOUT:
            lost_packets_ts.append(timestamp)
            del sent_packets[seq] # Remove from dict after counting as lost

    return rtt_records, lost_packets_ts

def resample_and_engineer_features(df, interval):
    """
    Resamples the dataframe and engineers advanced features.
    """
    df.set_index('timestamp', inplace=True)

    agg_rules = {
        'length': 'sum',
        'rtt': ['mean', 'min', 'max', 'std'],
        'is_from_laptop': 'count',
        'lost_packet': 'sum' # Sum of lost packets in the interval
    }

    resampled_df = df.resample(interval).agg(agg_rules)

    resampled_df.columns = ['_'.join(col).strip() for col in resampled_df.columns.values]
    resampled_df.rename(columns={
        'is_from_laptop_count': 'packet_count',
        'lost_packet_sum': 'lost_packet_count'
    }, inplace=True)

    resampled_df.fillna(0, inplace=True)

    resampled_df['rtt_mean_trend'] = resampled_df['rtt_mean'].diff().fillna(0)
    resampled_df['packet_count_trend'] = resampled_df['packet_count'].diff().fillna(0)
    resampled_df['length_sum_trend'] = resampled_df['length_sum'].diff().fillna(0)
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
        record = {'timestamp': pkt.time, 'length': len(pkt), 'is_from_laptop': 0}
        if IP in pkt and pkt[IP].src == laptop_ip:
            record['is_from_laptop'] = 1
        data.append(record)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').dropna()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    rtt_records, lost_packets_ts = analyze_tcp_flow(packets, laptop_ip)

    if rtt_records:
        rtt_df = pd.DataFrame(rtt_records)
        rtt_df['timestamp'] = pd.to_numeric(rtt_df['timestamp'], errors='coerce')
        rtt_df.dropna(subset=['timestamp'], inplace=True)
        rtt_df['timestamp'] = pd.to_datetime(rtt_df['timestamp'], unit='s')
        df = pd.merge_asof(df.sort_values('timestamp'), rtt_df.sort_values('timestamp'), on='timestamp', direction='backward')
        # Fill missing RTTs with a high value representing a timeout, instead of 0
        df['rtt'] = df['rtt'].ffill().fillna(PACKET_LOSS_TIMEOUT)
    else:
        # If there are no RTT records at all, all RTTs should be considered timed out
        df['rtt'] = PACKET_LOSS_TIMEOUT

    # Add a column for lost packets
    df['lost_packet'] = 0
    if lost_packets_ts:
        # For each lost packet, find the closest timestamp in the main df and mark it as lost
        numeric_lost_ts = pd.to_numeric(pd.Series(lost_packets_ts), errors='coerce').dropna()
        lost_times = pd.to_datetime(numeric_lost_ts, unit='s')
        # Use searchsorted to find insertion points, which corresponds to nearest index
        indices = df['timestamp'].searchsorted(lost_times)
        # To avoid index out of bounds
        valid_indices = [i for i in indices if i < len(df)]
        df.loc[valid_indices, 'lost_packet'] = 1


    engineered_df = resample_and_engineer_features(df, interval)

    return engineered_df
