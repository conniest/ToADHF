# cw_decoder.py

import numpy as np
import scipy.signal
import sounddevice as sd
import threading
import time

from cw_common import MORSE_CODE, SAMPLE_RATE, TONE_FREQ, WPM, DOT_DURATION

class CWDecoder:
    def __init__(self):
        self.buffer = []
        self.state = 0
        self.last_tone = time.time()
        self.symbol_buffer = ''

    def process_chunk(self, samples):
        # Bandpass filter to isolate CW tone
        b, a = scipy.signal.butter(4, [650 / (SAMPLE_RATE / 2), 750 / (SAMPLE_RATE / 2)], btype='band')
        filtered = scipy.signal.lfilter(b, a, samples)

        # Envelope detection
        envelope = np.abs(scipy.signal.hilbert(filtered))
        mean_level = np.mean(envelope)
        tone_present = envelope > (mean_level * 1.5)

        # Measure tone durations
        current_time = time.time()
        tone_active = np.mean(tone_present) > 0.5

        if tone_active:
            self.last_tone = current_time
            self.state += 1
        else:
            if self.state > 0:
                duration = self.state * len(samples) / SAMPLE_RATE
                if duration < DOT_DURATION * 1.5:
                    self.symbol_buffer += '.'
                else:
                    self.symbol_buffer += '-'
                self.state = 0

            if current_time - self.last_tone > DOT_DURATION * 3:
                self.flush_symbol()

    def flush_symbol(self):
        if self.symbol_buffer:
            char = MORSE_CODE.get(self.symbol_buffer, '?')
            print(char, end='', flush=True)
            self.symbol_buffer = ''


def start_cw_rx():
    decoder = CWDecoder()

    def callback(indata, frames, time_info, status):
        samples = indata[:, 0]
        decoder.process_chunk(samples)

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=callback, blocksize=int(SAMPLE_RATE * DOT_DURATION / 2)):
        print(f"[CW RX] Listening at {WPM} WPM... Press Ctrl+C to exit.")
        while True:
            time.sleep(0.1)

if __name__ == '__main__':
    start_cw_rx()
