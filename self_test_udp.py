import socket
import threading
import time

def receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", 4000))
        print("[TEST] Bound to 0.0.0.0:4000 successfully.")
        sock.settimeout(3.0)
        
        print("[TEST] Waiting for packet...")
        data, addr = sock.recvfrom(1024)
        print(f"[TEST] SUCCESS: Received '{data.decode()}' from {addr}")
        
    except Exception as e:
        print(f"[TEST] FAILURE: {e}")
    finally:
        sock.close()

def sender():
    time.sleep(1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        print(f"[TEST] Sending packet to 127.0.0.1:4000...")
        sock.sendto(b"HELLO_SELF", ("127.0.0.1", 4000))
        
        # Also try the actual IP to test routing
        my_ip = "172.20.64.244"
        print(f"[TEST] Sending packet to {my_ip}:4000...")
        sock.sendto(b"HELLO_IP", (my_ip, 4000))
        
    except Exception as e:
        print(f"[TEST] Sender Error: {e}")

if __name__ == "__main__":
    t = threading.Thread(target=receiver)
    t.start()
    sender()
    t.join()
