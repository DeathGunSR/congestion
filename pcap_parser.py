from scapy.all import rdpcap
from feature_extractor import process_packets

# --- Configuration ---
PCAP_FILE = 'training_traffic.pcap' # Default file for training
PROCESSED_CSV_FILE = 'processed_data.csv'
LAPTOP_IP = '192.168.1.100' # Default laptop IP
TIME_INTERVAL = '5S'


def main():
    """
    Main function to parse a pcap file and save the processed data.
    """
    print(f"Reading pcap file: {PCAP_FILE}...")
    try:
        packets = rdpcap(PCAP_FILE)
    except FileNotFoundError:
        print(f"Error: The file '{PCAP_FILE}' was not found.")
        print("Please make sure the pcap file is in the same directory and the filename is correct.")
        return

    print(f"Total packets read: {len(packets)}")

    laptop_ip_address = input(f"Enter the laptop's IP address (default: {LAPTOP_IP}): ")
    if not laptop_ip_address:
        laptop_ip_address = LAPTOP_IP

    # Use the centralized processing function
    aggregated_df = process_packets(packets, laptop_ip_address, TIME_INTERVAL)

    if aggregated_df.empty:
        print("No data was processed. The output file will not be created.")
        return

    # --- Save to CSV ---
    aggregated_df.to_csv(PROCESSED_CSV_FILE, index=False)
    print(f"Aggregated data saved to {PROCESSED_CSV_FILE}")

if __name__ == '__main__':
    main()
    print("\nScript finished.")
