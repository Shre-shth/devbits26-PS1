import socket
import subprocess
import struct

ASTERISK_IP = "172.20.30.87"
ASTERISK_PORT = 4000
LISTEN_PORT = 4000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", LISTEN_PORT))

print("Listening for RTP...")

def send_rtp(payload, seq, timestamp):
    header = struct.pack("!BBHII",
        0x80,      # Version
        118,       # Payload type
        seq,
        timestamp,
        12345      # SSRC
    )
    sock.sendto(header + payload, (ASTERISK_IP, ASTERISK_PORT))

seq = 0
timestamp = 0

while True:
    data, addr = sock.recvfrom(2048)

    # Remove RTP header (12 bytes)
    pcm = data[12:]

    # ---- SEND TO WHISPER HERE ----
    # For now, echo back audio

    send_rtp(pcm, seq, timestamp)

    seq += 1
    timestamp += 320
