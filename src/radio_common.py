# radio_common.py

import subprocess
import threading

class IC7300:
    def __init__(self, device="/dev/ttyUSB0", audio="USB Audio CODEC", model=3073, baud=115200):
        self.device = device
        self.model = model
        self.baud = baud
        self.tx_lock = threading.Lock()
        self.audio = audio

    def _rigctl(self, *args):
        cmd = [
            "rigctl",
            "-m", str(self.model),
            "-r", self.device,
            "-s", str(self.baud),
        ]
        cmd.extend(*args)
        print(cmd)
        #print(f"Running command: {' '.join(c for c in cmd)}")
        subprocess.run(cmd, check=True)

    def enable_sidetone(self, level=1.0):
        """Enable monitor audio for sidetone generation during PTT."""
        self._rigctl(["l", "MONITOR_GAIN", str(level)])

    def set_freq(self, freq_hz):
        """Set operating frequency."""
        self._rigctl(["F", str(freq_hz)])

    def set_mode(self, mode):
        """Set mode and passband."""
        mode_map = {
            # Sideband
            "USB": ("USB", "2400"),
            "LSB": ("LSB", "2400"),
            # Data
            "DATA-U": ("USB-D", "3000"),
            "DATA": ("USB-D", "3000"),
            "DATA-L": ("LSB-D", "3000"),
            # CW 
            "CW": ("CW", "500")
        }
        if mode not in mode_map:
            raise ValueError(f"Unsupported mode: {mode}")
        cmd = ["M"] + [m for m in mode_map[mode]]
        self._rigctl(cmd)

    def ptt_on(self):
        self._rigctl(["T", "1"])

    def ptt_off(self):
        self._rigctl(["T", "0"])

    def close(self):
        pass  # rigctl does not require closing

    def transmit_audio_block(self, audio_callback):
        """Key PTT, call audio_callback() to play audio, then unkey."""
        with self.tx_lock:
            self.ptt_on()
            audio_callback()
            self.ptt_off()

    def __repr__(self):
        return f"<IC 7300 port={self.device} audio={self.audio}>"

class MockIC7300:
    def __init__(self, device="/dev/cu.SLAB_USBtoUART", model=3073, baud=115200):
        self.device = device
        self.model = model
        self.baud = baud
        self.tx_lock = threading.Lock()
        self.commands = []  # Store issued commands for inspection

    def _rigctl(self, *args):
        cmd = [
            "rigctl",
            "-m", str(self.model),
            "-r", self.device,
            "-s", str(self.baud),
            *args
        ]
        print(f"[MOCK] Would run: {' '.join(cmd)}")
        self.commands.append(cmd)

    def enable_sidetone(self, level=1.0):
        self._rigctl("l", "MONITOR_GAIN", str(level))

    def set_freq(self, freq_hz):
        self._rigctl("F", str(freq_hz))

    def set_mode(self, mode):
        mode_map = {
            "USB": ("USB", "2400"),
            "CW": ("CW", "500")
        }
        if mode not in mode_map:
            raise ValueError(f"Unsupported mode: {mode}")
        mode_str, passband = mode_map[mode]
        self.set_freq(14070000)
        self._rigctl("M", mode_str, passband)

    def ptt_on(self):
        self._rigctl("T", "1")

    def ptt_off(self):
        self._rigctl("T", "0")

    def close(self):
        print("[MOCK] Closing mock rig interface.")

    def transmit_audio_block(self, audio_callback):
        with self.tx_lock:
            self.ptt_on()
            print("[MOCK] Playing audio...")
            audio_callback()
            self.ptt_off()


class K3S:
    def __init__(self, device="/dev/ttyUSB0", model=2043, baud=38400):
        self.device = device
        self.model = model  # 229 is the Hamlib model for Elecraft K3/K3S
        self.baud = baud
        self.tx_lock = threading.Lock()

    def _rigctl(self, *args):
        cmd = [
            "rigctl",
            "-m", str(self.model),
            "-r", self.device,
            "-s", str(self.baud),
            *args
        ]
        subprocess.run(cmd, check=True)

    def enable_sidetone(self, level=1.0):
        """Enable sidetone or monitor audio if available (Elecraft may not support this directly)."""
        try:
            self._rigctl("l", "MONITOR_GAIN", str(level))
        except subprocess.CalledProcessError:
            print("Monitor gain setting not supported on K3S.")

    def set_freq(self, freq_hz):
        """Set operating frequency."""
        self._rigctl("F", str(freq_hz))

    def set_mode(self, mode):
        """Set mode and bandwidth if supported."""
        mode_map = {
            "USB": ("USB", "2400"),
            "LSB": ("LSB", "2400"),
            "CW": ("CW", "500"),
            "CW-R": ("CW-R", "500"),
            "RTTY": ("RTTY", "300"),
            "DATA": ("DATA", "2400"),
        }
        if mode not in mode_map:
            raise ValueError(f"Unsupported mode: {mode}")

        self._rigctl("M", mode_map[mode][0])
        # Elecraft K3 supports custom filter widths, but setting bandwidth may not work with rigctl

    def ptt_on(self):
        self._rigctl("T", "1")

    def ptt_off(self):
        self._rigctl("T", "0")

    def close(self):
        pass  # Nothing to close for rigctl

    def transmit_audio_block(self, audio_callback):
        """Key PTT, call audio_callback() to play audio, then unkey."""
        with self.tx_lock:
            self.ptt_on()
            audio_callback()
            self.ptt_off()

