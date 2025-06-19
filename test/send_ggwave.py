import ggwave
import sounddevice as sd
import numpy as np

def send_ggwave(message: str, protocol_id=1, volume=20, rate=48000):
    """
    Transmit a GGWave-encoded message using sounddevice.

    Parameters:
        message (str): Text message to encode and send.
        protocol_id (int): GGWave protocol ID (e.g. 1 = AUDIBLE_FAST).
        volume (int): Volume scaling (0–100).
        rate (int): Sample rate in Hz (typically 48000).
    """
    # Encode message to waveform (float32 PCM bytes)
    waveform_bytes = ggwave.encode(message, protocolId=protocol_id, volume=volume)
    
    # Convert to NumPy float32 array for sounddevice
    samples = np.frombuffer(waveform_bytes, dtype=np.float32)

    print(f"Transmitting '{message}' via GGWave...")
    sd.play(samples, samplerate=rate)
    sd.wait()


send_ggwave("CQ CQ DE KN6UBF")
