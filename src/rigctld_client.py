# rigctld_client.py
# Start rigctld with rigctld -m 3073 -r /dev/cu.SLAB_USBtoUART -s 115200 &
import socket
import threading
import time

class RigctldClient:
    def __init__(self, host="localhost", port=4532):
        self.host = host
        self.port = port
        self.tx_lock = threading.Lock()
        self.sock = socket.create_connection((host, port))
        self.sock_file = self.sock.makefile("rw")

    def send_command(self, cmd):
        self.sock_file.write(cmd.strip() + "\n")
        self.sock_file.flush()
        return self.sock_file.readline().strip()

    def set_freq(self, freq_hz):
        return self.send_command(f"F {freq_hz}")

    def set_mode(self, mode, passband):
        return self.send_command(f"M {mode} {passband}")

    def ptt_on(self):
        return self.send_command("T 1")

    def ptt_off(self):
        return self.send_command("T 0")

    def send_morse(self, message):
        return self.send_command(f"SEND_MORSE {message}")

    def set_monitor(self, level):
        return self.send_command(f"l MON {level}")

    def close(self):
        self.sock_file.close()
        self.sock.close()

if __name__ == '__main__':
    rig = RigctldClient()
    print(rig.set_freq(14070000))
    print(rig.set_mode("CW", 500))
    print(rig.send_morse("CQ CQ DE KN6UBF"))
    rig.close()
