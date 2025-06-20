import sounddevice as sd
import numpy as np
import collections
import soundfile as sf
import ggwave
import time
from datetime import datetime

ctx = ggwave.init()

# === Config ===
device_name = "USB Audio CODEC"  # Replace with your actual USB audio input
samplerate = 48000
channels = 1
buffer_time = 0.5  # seconds
samples_needed = int(buffer_time * samplerate)
buffer = collections.deque(maxlen=samples_needed)
last_decode_time = 0
decode_interval = 0.5  # seconds between decode attempts

print(f"[GGWave] Listening on: {device_name} | Buffer: {buffer_time:.2f}s")

# === Callback ===
def callback(indata, frames, time_info, status):
    global last_decode_time

    buffer.extend(indata[:, 0])
    now = time.time()

    # Only decode every N seconds AND when buffer is full
    if len(buffer) >= samples_needed and (now - last_decode_time) >= decode_interval:
        last_decode_time = now

        pcm_array = np.array(buffer, dtype=np.float32)
        pcm_bytes = pcm_array.tobytes()

        # Attempt GGWave decode
        result = ggwave.decode(ctx,pcm_bytes)

        # Save buffer for debugging
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"ggwave_buffer_{ts}.wav"
        sf.write(filename, pcm_array, samplerate)
        print(f"[DEBUG] Saved buffer to {filename}")

        if result:
            print(f"\n[RECV] {result.decode()}\n> ", end='', flush=True)
            buffer.clear()
        else:
            print(".", end='', flush=True)

# === Start Listening ===
with sd.InputStream(
    device=device_name,
    channels=channels,
    samplerate=samplerate,
    callback=callback
):
    while True:
        time.sleep(0.1)
