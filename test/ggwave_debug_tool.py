import sounddevice as sd
import soundfile as sf
import numpy as np
import ggwave
import time
import datetime
import threading

try:
    from radio_common import IC7300
    use_cat = True
except ImportError:
    IC7300 = None
    use_cat = False

# === CONFIG ===
message = "CQ CQ DE KN6UBF TOAD TEST"
protocol_id = 0
samplerate = 48000
usb_audio_input = "USB Audio CODEC"
usb_audio_output = "USB Audio CODEC"
record_duration = 5.0  # seconds
ptt_enabled = use_cat
output_filename = f"ggwave_rx_{datetime.datetime.now().strftime('%H%M%S')}.wav"

# === GGWave Payload ===
ctx = ggwave.init()
payload = ggwave.encode(ctx, message.encode(), protocolId=protocol_id)
samples = np.frombuffer(payload, dtype=np.float32)
padding = np.zeros(int(0.5 * samplerate), dtype=np.float32)
padded = np.concatenate([padding, samples, padding])
print(f"[INFO] Payload length: {len(padded)/samplerate:.2f} sec")

# === Radio Setup (Optional) ===
radio = IC7300() if (ptt_enabled and IC7300) else None
if radio:
    radio.set_mode("USB-D")

# === Recording Buffer ===
recorded_chunks = []

def start_recording(stream, done_event):
    def callback(indata, frames, time_info, status):
        recorded_chunks.append(indata.copy())

    stream.start()
    stream.read_available  # kick things off
    print(f"[INFO] Recording from '{usb_audio_input}'...")
    stream.read(1)  # prime stream
    stream.callback = callback
    done_event.wait()
    stream.stop()
    stream.close()

recording_done = threading.Event()
stream = sd.InputStream(device=usb_audio_input, channels=1, samplerate=samplerate)
record_thread = threading.Thread(target=start_recording, args=(stream, recording_done))
record_thread.start()

# === Transmit in Parallel ===
time.sleep(0.25)
if radio:
    radio.ptt_on()
    print("[INFO] PTT ON")

print("[INFO] Transmitting GGWave...")
sd.play(padded, samplerate=samplerate, device=usb_audio_output)
sd.wait()
print("[INFO] GGWave TX done.")

time.sleep(0.5)
if radio:
    radio.ptt_off()
    print("[INFO] PTT OFF")

# === Stop Recording ===
time.sleep(record_duration)
recording_done.set()
record_thread.join()
print("[INFO] Recording complete.")

# === Save and Decode ===
recorded = np.concatenate(recorded_chunks)
sf.write(output_filename, recorded, samplerate)
print(f"[INFO] Saved to {output_filename}")

pcm = recorded.astype(np.float32).tobytes()
result = ggwave.decode(ctx, pcm)
if result:
    print(f"[✅ DECODED] {result.decode()}")
else:
    print("[❌ DECODE FAILED]")
