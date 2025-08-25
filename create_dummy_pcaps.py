import time
from scapy.all import *

# --- Configuration ---
CLIENT_IP = "192.168.1.100" # Same as LAPTOP_IP in pcap_parser
SERVER_IP = "8.8.8.8"
CLIENT_PORT = 12345
SERVER_PORT = 443 # HTTPS
OUTPUT_DIR = "data"

def create_packet_flow(start_time, num_packets, rtt_generator, loss_indices=None):
    """
    Generates a list of request-response packet pairs.
    - rtt_generator: A function that returns the RTT for the next packet.
    - loss_indices: A set of indices where the response packet should be dropped.
    """
    packets = []
    current_time = start_time
    if loss_indices is None:
        loss_indices = set()

    for i in range(num_packets):
        # Client to Server (Request)
        req = IP(src=CLIENT_IP, dst=SERVER_IP) / TCP(sport=CLIENT_PORT, dport=SERVER_PORT, flags='S', seq=i)
        req.time = current_time
        packets.append(req)

        # Server to Client (Response) - maybe lost
        if i not in loss_indices:
            rtt = rtt_generator()
            resp_time = current_time + rtt
            resp = IP(src=SERVER_IP, dst=CLIENT_IP) / TCP(sport=SERVER_PORT, dport=CLIENT_PORT, flags='SA', ack=i + 1)
            resp.time = resp_time
            packets.append(resp)

        # Move time forward for the next request
        current_time += 0.05 # 50ms between requests
    return packets

def generate_browsing_pcap():
    """Generates a stable browsing session."""
    print("Generating stable 'browsing' pcap...")
    start_time = time.time()
    # Low, stable RTT
    rtt_gen = lambda: random.uniform(0.03, 0.05) # 30-50ms RTT
    packets = create_packet_flow(start_time, 200, rtt_gen)
    wrpcap(os.path.join(OUTPUT_DIR, "browsing_stable.pcap"), packets)
    print("... 'browsing_stable.pcap' created.")

def generate_gaming_pcap():
    """Generates an unstable gaming session with RTT spikes and packet loss."""
    print("Generating unstable 'gaming' pcap...")
    start_time = time.time()

    # RTT generator with spikes
    def rtt_gen_spiky():
        if random.random() < 0.1: # 10% chance of a spike
            return random.uniform(0.2, 0.4) # 200-400ms RTT spike
        return random.uniform(0.04, 0.06) # 40-60ms normal RTT

    # Drop 5% of packets
    loss_indices = set(random.sample(range(200), k=10))
    print(f"Simulating packet loss at indices: {loss_indices}")
    packets = create_packet_flow(start_time, 200, rtt_gen_spiky, loss_indices)
    wrpcap(os.path.join(OUTPUT_DIR, "gaming_unstable.pcap"), packets)
    print("... 'gaming_unstable.pcap' created.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    generate_browsing_pcap()
    generate_gaming_pcap()
    print("\nDummy pcap files generated successfully.")
