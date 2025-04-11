# radio_common.py
import serial
import threading
import time

class IC7300:
    def __init__(self, port="/dev/ttyUSB0", baudrate=19200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.tx_lock = threading.Lock()

    def send_cat(self, command_bytes):
        self.ser.write(command_bytes)
        return self.ser.read(100)

    def set_mode(self, mode):
        MODES = {
            'USB': b'\x01\x04\x01\x00',
            'CW':  b'\x01\x04\x03\x00',
        }
        cmd = b'\xfe\xfe\x94\xe0' + MODES[mode] + b'\xfd'
        return self.send_cat(cmd)

    def ptt_on(self):
        return self.send_cat(b'\xfe\xfe\x94\xe0\x1c\x00\x01\xfd')

    def ptt_off(self):
        return self.send_cat(b'\xfe\xfe\x94\xe0\x1c\x00\x00\xfd')

    def close(self):
        self.ser.close()
