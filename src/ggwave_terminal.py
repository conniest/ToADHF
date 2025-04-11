# ggwave_terminal.py
from radio_common import IC7300
import ggwave
import threading
import sounddevice as sd
import numpy as np
import time

ctx = ggwave.init()

def listen_loop(radio: IC7300):
    def callback(indata, frames, time_info, status):
        if radio.tx_lock.locked():
            return  # skip decoding while transmitting
        pcm = indata[:, 0].copy()
        result = ggwave.decode(ctx, pcm)
        if result:
            print(f"\n[RECV] {result.decode()}\n> ", end='', flush=True)

    with sd.InputStream(channels=1, samplerate=48000, callback=callback):
        while True:
            time.sleep(0.1)

def main():
    radio = IC7300()
    radio.set_mode('USB')

    rx_thread = threading.Thread(target=listen_loop, args=(radio,), daemon=True)
    rx_thread.start()

    try:
        while True:
            text = input("> ")
            payload = ggwave.encode(ctx, text.encode(), protocolId=1)
            with radio.tx_lock:
                radio.ptt_on()
                sd.play(payload, samplerate=48000)
                sd.wait()
                radio.ptt_off()
    finally:
        radio.close()

if __name__ == '__main__':
    main()
