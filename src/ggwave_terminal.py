# ggwave_terminal.py
from radio_common import IC7300
import ggwave
import threading
import sounddevice as sd
import numpy as np
import time

ctx = ggwave.init()
usb_audio_output_device = "USB Audio CODEC"

def listen_loop(radio: IC7300):
    def callback(indata, frames, time_info, status):
        if radio.tx_lock.locked():
            return  # skip decoding while transmitting
        pcm = indata[:, 0].copy().tobytes()
        result = ggwave.decode(ctx, pcm)
        if result:
            print(f"\n[RECV] {result.decode()}\n> ", end='', flush=True)

    with sd.InputStream(channels=1, samplerate=48000, callback=callback):
        while True:
            time.sleep(0.1)

def main():
    radio = IC7300()
    radio.set_mode('LSB')

    # Optional: set sounddevice defaults
    # sd.default.device = (None, usb_audio_output_device)  # (output, input)

    rx_thread = threading.Thread(target=listen_loop, args=(radio,), daemon=True)
    rx_thread.start()

    try:
        while True:
            text = input("> ")
            payload = ggwave.encode( text.encode(), protocolId=1)
            with radio.tx_lock:
                radio.ptt_on()
                samples = np.frombuffer(payload, dtype=np.float32)
                sd.play(samples, samplerate=48000, device=2)
                sd.wait()
                radio.ptt_off()
    finally:
        radio.close()

if __name__ == '__main__':
    main()
