# cw_terminal.py

from radio_common import IC7300
from cw_common import MORSE_CODE, DOT_DURATION, SAMPLE_RATE
import sounddevice as sd
import threading
import numpy as np
import time

CW_TABLE = {v: k for k, v in MORSE_CODE.items()}  # Reverse lookup

def send_cw(radio, message):
    for char in message.upper():
        if char == ' ':
            time.sleep(DOT_DURATION * 7)  # Space between words
        elif char in CW_TABLE:
            for symbol in CW_TABLE[char]:
                radio.ptt_on()
                time.sleep(DOT_DURATION if symbol == '.' else DOT_DURATION * 3)
                radio.ptt_off()
                time.sleep(DOT_DURATION)  # Gap between symbols
            time.sleep(DOT_DURATION * 3)  # Gap between letters

def cw_rx_loop(radio: IC7300):
    from cw_decoder import CWDecoder
    decoder = CWDecoder()

    def callback(indata, frames, time_info, status):
        if radio.tx_lock.locked():
            return
        samples = indata[:, 0]
        decoder.process_chunk(samples)

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=callback, blocksize=int(SAMPLE_RATE * DOT_DURATION / 2)):
        while True:
            time.sleep(0.1)

def main():
    radio = IC7300()
    radio.set_mode('CW')

    rx_thread = threading.Thread(target=cw_rx_loop, args=(radio,), daemon=True)
    rx_thread.start()

    try:
        while True:
            text = input("> ")
            with radio.tx_lock:
                send_cw(radio, text)
    finally:
        radio.close()

if __name__ == '__main__':
    main()
