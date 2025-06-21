import soundfile as sf
import ggwave
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter, filtfilt

def bandpass_filter(data, lowcut=300.0, highcut=3500.0, fs=48000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return lfilter(b, a, data)

def bandpass_zero_phase(data, lowcut=300, highcut=4500, fs=48000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def lowpass_filter(data, cutoff=4500.0, fs=48000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, data)

def rms_normalize(signal, target=0.9):
    rms = np.sqrt(np.mean(signal**2))
    if rms == 0:
        return signal
    return signal * (target / rms)

def apply_window(signal):
    window = np.hanning(len(signal))
    return signal * window

def decode_wav_snippet(path, start_s=3.3, end_s=4.83, protocol_id=1):
    params = ggwave.getDefaultParameters()
    params["sampleRateInp"] = 48000.0
    params["sampleRateOut"] = 48000.0
    params["sampleRate"] = 48000.0
    params["samplesPerFrame"] = 1024
    params["operatingMode"] = 3
    params["soundMarkerThreshold"] = 1.0
    params["sampleFormatInp"] = 5
    params["sampleFormatOut"] = 5
    params["payloadLength"] = -1
    ctx = ggwave.init(params)

    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data[:, 0]

    pad = int(0.2 * sr)  # 200 ms pre/post padding
    start_idx = max(0, int(start_s * sr) - pad)
    end_idx = min(len(data), int(end_s * sr) + pad)
    snippet = data[start_idx:end_idx]

    # New signal conditioning
    filtered = bandpass_zero_phase(snippet, fs=sr)
    normalized = rms_normalize(filtered, target=0.5)

    # Apply fade-in/fade-out only
    fade_len = int(0.05 * len(normalized))
    window = np.ones(len(normalized))
    window[:fade_len] = np.hanning(fade_len * 2)[:fade_len]
    window[-fade_len:] = np.hanning(fade_len * 2)[-fade_len:]
    final = normalized * window


    pcm = rms_normalize(final.astype(np.float32), target=0.15).astype(np.float32)

    # 🔊 Playback
    print("[▶️  Playing snippet...]")
    sd.play(pcm, samplerate=sr)
    sd.wait()

    # Save processed sample
    sf.write("recordings/test.wav", pcm, sr)

    # Write known-good reference GGWave encoding
    encoded = ggwave.encode("kn6ubf")
    samples = np.frombuffer(encoded, dtype=np.float32)
    sf.write("recordings/from_encode.wav", samples, 48000)


    # Decode
    print(f"[DEBUG] Feeding {len(pcm)} samples, max amp = {np.max(np.abs(pcm)):.4f}")
    result = ggwave.decode(ctx, pcm.tobytes())
    if result:
        print(f"[✅ DECODED] {result.decode()}")
    else:
        print("[❌ DECODE FAILED]")

    ctx = ggwave.init(params)
    with sf.SoundFile("recordings/test.wav") as f:
        pcm = f.read(dtype='float32')
    result = ggwave.decode(ctx, pcm.tobytes())
    print(result)

# Run
decode_wav_snippet("recordings/toad_20250620_193950.wav")
