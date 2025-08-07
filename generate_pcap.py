import random
from scapy.all import *
import time
import os

# --- Configuration ---
FILENAME = "traffic.pcap"
TOTAL_DURATION = 600  # 10 minutes in seconds
CONGESTION_PERIODS = [(120, 270), (490, 540)]  # As specified by the user
BASE_RATE_PPS = 10  # Packets per second during normal traffic
CONGESTION_RATE_PPS = 50  # Packets per second during congestion

# --- Helper Functions ---

def random_mac():
    return "02:00:00:%02x:%02x:%02x" % (random.randint(0, 255),
                                     random.randint(0, 255),
                                     random.randint(0, 255))

def is_congested(current_time):
    for start, end in CONGESTION_PERIODS:
        if start <= current_time < end:
            return True
    return False

def create_wifi_packet(src_mac, ap_mac="00:11:22:33:44:55"):
    """Creates a random type of WiFi packet."""
    p_type = random.choices([0, 1, 2], weights=[0.4, 0.1, 0.5])[0] # Management, Control, Data

    # More management frames during congestion
    if is_congested(current_time):
        p_type = random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0]

    addr1 = ap_mac  # Destination
    addr2 = src_mac   # Source
    addr3 = ap_mac    # BSSID

    if p_type == 0:  # Management Frame
        subtype = random.choice([0, 4, 8]) # Assoc Req, Probe Req, Beacon
        p = RadioTap() / Dot11(type="Management", subtype=subtype, addr1=addr1, addr2=addr2, addr3=addr3)
    elif p_type == 1:  # Control Frame
        subtype = random.choice([11, 12]) # RTS, CTS
        p = RadioTap() / Dot11(type="Control", subtype=subtype, addr1=addr1)
    else:  # Data Frame
        p = RadioTap() / Dot11(type="Data", addr1=addr1, addr2=addr2, addr3=addr3) / LLC() / SNAP() / IP(src="192.168.1.%d" % random.randint(100,200), dst="8.8.8.8") / TCP() / Raw(b"X"*random.randint(50, 1200))

    return p

# --- Main Generation Logic ---

if os.path.exists(FILENAME):
    print(f"File '{FILENAME}' already exists. Deleting it.")
    os.remove(FILENAME)

print(f"Generating synthetic pcap file: {FILENAME}")
print(f"Total duration: {TOTAL_DURATION} seconds")

# Use a fixed start time for reproducibility
current_time = time.time()
pcap_writer = PcapWriter(FILENAME, append=False, sync=True)

# Generate a pool of devices
device_macs = [random_mac() for _ in range(100)]
ap_mac = "00:11:22:33:44:55"

for t in range(TOTAL_DURATION):
    if is_congested(t):
        rate = CONGESTION_RATE_PPS
        active_devices = int(len(device_macs) * 0.8) # 80% of devices are active
    else:
        rate = BASE_RATE_PPS
        active_devices = int(len(device_macs) * 0.2) # 20% of devices are active

    # Get a sample of devices for this second
    current_active_macs = random.sample(device_macs, active_devices)

    for _ in range(rate):
        src = random.choice(current_active_macs)
        packet = create_wifi_packet(src, ap_mac)

        # Add a small time jitter
        packet.time = current_time + (random.random() * 0.9)
        pcap_writer.write(packet)

    current_time += 1
    if t % 60 == 0:
        print(f"  ... simulated {t//60} minutes ...")

pcap_writer.close()

print("Synthetic pcap file generation complete.")
