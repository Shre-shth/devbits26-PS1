#!/usr/bin/env python3

import socket
import subprocess

ASTERISK_IP = "172.20.50.32"
ASTERISK_PORT = 4000
LOCAL_PORT = 4000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", LOCAL_PORT))

print("Listening for RTP on port 4000...")

def send_audio_back(audio_bytes):
    # RTP header (PT=96 dynamic)
    rtp_header = b"\x80\x60\x00\x01\x00\x00\x00\x01\x12\x34\x56\x78"
    sock.sendto(rtp_header + audio_bytes, (ASTERISK_IP, ASTERISK_PORT))

while True:
    data, addr = sock.recvfrom(2048)

    # Skip RTP header (12 bytes)
    pcm = data[12:]

    print(f"Received {len(pcm)} bytes of audio")

    # Dummy reply (replace with TTS later)
    send_audio_back(pcm)
