import ggwave
import sounddevice as sd
import numpy as np
import time

# Set this to your IC-7300's USB audio *output* device (from `sd.query_devices()`)
usb_audio_device = "USB Audio CODEC"  # Replace with actual name or index if needed

# GGWave message
message = "CQ CQ DE KN6UBF TOADHF"
protocol_id = 1  # AUDIBLE_FAST
volume = 100     # Max volume for strong signal
samplerate = 48000

# Encode GGWave signal
payload = ggwave.encode(message, protocolId=protocol_id, volume=volume)
samples = np.frombuffer(payload, dtype=np.float32)

print(f"Sending GGWave message into IC-7300: \"{message}\"")

# Optional: pause to start listener
time.sleep(1.0)

# Play into rig
sd.play(samples, samplerate=samplerate, device=usb_audio_device)
sd.wait()

print("Transmission complete.")
