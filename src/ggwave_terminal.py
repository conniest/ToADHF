# ggwave_terminal.py
from radio_common import IC7300
import ggwave
import threading
import sounddevice as sd
import numpy as np
import time
import collections

ctx = ggwave.init()
usb_audio_output_device = "USB Audio CODEC"

def listen_loop(radio, device="USB Audio CODEC", samplerate=48000):
    frame_duration = 10  # GGWave frame ~300ms, so 0.5s gives margin
    step_fraction = 0.2   # Slide forward 20% each decode
    buffer = collections.deque()
    samples_per_frame = int(frame_duration * samplerate)
    step_size = int(samples_per_frame * step_fraction)

    print(f"[GGWave] Listening on: {device} | Window: {frame_duration:.2f}s")

    def callback(indata, frames, time_info, status):
        if radio.tx_lock.locked(): return
        buffer.extend(indata[:, 0])

        # Only decode when we have enough samples
        while len(buffer) >= samples_per_frame:
            pcm = np.array(list(buffer)[:samples_per_frame], dtype=np.float32).tobytes()
            result = ggwave.decode(ctx, pcm)

            if result:
                print(f"\n[RECV] {result.decode()}\n> ", end='', flush=True)
                #buffer.clear()
                return
            else:
                # Slide window forward
                for _ in range(step_size):
                    buffer.popleft()

    with sd.InputStream(device=device, channels=1, samplerate=samplerate, callback=callback):
        while True:
            time.sleep(0.1)
def main():
    radio = IC7300()
    radio.set_mode('LSB-D')

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
                time.sleep(.5)
                samples = np.frombuffer(payload, dtype=np.float32)
                padding = np.zeros(int(.02 * 48000), dtype=np.float32)  # 250ms silence
                padded = np.concatenate([padding, samples, padding])
                sd.play(padded, samplerate=48000, device=2)
                time.sleep(0.25)
                sd.wait()
                time.sleep(.5)
                radio.ptt_off()
    finally:
        radio.close()

if __name__ == '__main__':
    main()
