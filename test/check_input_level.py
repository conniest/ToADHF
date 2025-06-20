import sounddevice as sd
import numpy as np

def check_input_level(device):
    with sd.InputStream(device=device, channels=1, samplerate=48000) as stream:
        print("Measuring input volume...")
        for _ in range(5000):
            data, _ = stream.read(1024)
            level = np.abs(data).mean()
            print(f"Volume: {level:.6f}")

check_input_level(2)