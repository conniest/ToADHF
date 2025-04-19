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
            time.sleep(DOT_DURATION * 7)  # inter-word space
            continue

        morse = CW_TABLE.get(char)
        if not morse:
            continue

        for i, symbol in enumerate(morse):
            t_1 = time.time()
            radio.ptt_on()
            t_2 = time.time()
            print(t_2 - t_1)
            time.sleep(DOT_DURATION if symbol == '.' else DOT_DURATION * 3)
            t_3 = time.time()
            radio.ptt_off()
            t_4 = time.time()
            print(t_4 - t_3)

            # inter-element space (not after final element)
            if i < len(morse) - 1:
                time.sleep(DOT_DURATION)

        # inter-character space
        time.sleep(DOT_DURATION * 3)


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
