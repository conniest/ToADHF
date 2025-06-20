# ggwave_terminal.py
from radio_common import IC7300
import ggwave
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import collections
import os
import datetime
from scipy.signal import butter, lfilter, spectrogram

ctx = ggwave.init()
usb_audio_output_device = "USB Audio CODEC"

import numpy as np



def detect_sfd_pair_region(data, fs, band=(350, 3500),
                           baseline_window_sec=2.0,
                           min_duration_sec=0.1,
                           min_rise_db=10,
                           pad_sec=0.1):
    """
    Detect the first pair of GGWave SFD bursts and return the full region between them.
    Applies adaptive power rise detection over a specified frequency band.
    """
    nperseg = 1024
    noverlap = nperseg // 2
    f, t, Sxx = spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)

    f_mask = (f >= band[0]) & (f <= band[1])
    band_power = Sxx[f_mask].mean(axis=0)
    band_power_db = 10 * np.log10(band_power + 1e-10)

    baseline_win = int((baseline_window_sec * fs) / (nperseg - noverlap))
    smoothed_baseline = np.convolve(band_power_db, np.ones(baseline_win) / baseline_win, mode='same')
    delta_db = band_power_db - smoothed_baseline

    min_frames = int((min_duration_sec / (nperseg / fs)) + 0.5)
    above_delta = delta_db > min_rise_db

    # Find all SFD-like regions
    sfd_times = []
    count = 0
    for i, val in enumerate(above_delta):
        if val:
            count += 1
            if count == min_frames:
                start_time = t[i - count + 1]
        else:
            if count >= min_frames:
                end_time = t[i]
                sfd_times.append((start_time, end_time))
            count = 0

    if count >= min_frames:
        end_time = t[-1]
        sfd_times.append((start_time, end_time))

    if len(sfd_times) >= 2:
        s1_start, _ = sfd_times[0]
        _, s2_end = sfd_times[1]
        start_sample = max(0, int((s1_start - pad_sec) * fs))
        end_sample = min(len(data), int((s2_end + pad_sec) * fs))
        return start_sample, end_sample

    return None, None

def bandpass_filter(data, lowcut=350.0, highcut=3500.0, fs=48000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, data)

def detect_ggwave_region(data, fs, window_ms=100, threshold=0.005):
    win = int((window_ms / 1000) * fs)
    step = win // 2
    band_data = bandpass_filter(data, fs=fs)

    in_region = False
    start = None
    end = None

    for i in range(0, len(data) - win, step):
        window = band_data[i:i+win]
        rms = np.sqrt(np.mean(window**2))

        if rms > threshold:
            if not in_region:
                start = max(0, i - fs // 10)  # 100ms pad before start
                in_region = True
            end = i + win
        elif in_region:
            end = min(len(data), end + fs // 10)  # 100ms pad after end
            return start, end

    if in_region:
        end = min(len(data), end + fs // 10)
        return start, end

    return None, None

def listen_loop(radio, device="USB Audio CODEC", samplerate=48000):
    record_duration = 10.0  # seconds per chunk
    output_dir = "recordings"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[ToAD] Listening on '{device}' – saving {record_duration}s chunks to disk")

    frames_per_chunk = int(record_duration * samplerate)
    stream = sd.InputStream(device=device, channels=1, samplerate=samplerate, dtype='float32')
    stream.start()

    while True:
        if radio.tx_lock.locked():
            time.sleep(0.1)
            continue

        print("[ToAD] Capturing audio chunk...")
        audio, _ = stream.read(frames_per_chunk)

        filename = os.path.join(
            output_dir,
            f"toad_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        sf.write(filename, audio, samplerate)
        print(f"[ToAD] Saved → {filename}")

        result = None
        pcm = audio[:, 0].astype(np.float32)

        # Use filtered audio for detection, unfiltered for decode
        filtered = bandpass_filter(pcm, fs=samplerate)

        start, end = detect_sfd_pair_region(filtered, samplerate)
        if start is not None and end is not None:
            print(f"[ToAD] Detected GGWave burst from {start/samplerate:.2f}s to {end/samplerate:.2f}s")
            burst = pcm[start:end]
            pcm_bytes = burst.astype(np.float32).tobytes()
            result = ggwave.decode(ctx, pcm_bytes)
        else:
            print(f"[ToAD] No GGWave SFD detected in {filename}")

        if result:
            print(f"\n[RECV] {result.decode()}\n> ", end='', flush=True)
        else:
            print(f"[ToAD] No decode from {filename}")

def main():
    radio = IC7300()
    radio.set_mode('USB-D')

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
                samples = np.frombuffer(payload, dtype=np.float32)
                padding = np.zeros(int(.02 * 48000), dtype=np.float32)  # 250ms silence
                padded = np.concatenate([padding, samples, padding])
                sd.play(padded, samplerate=48000, device=usb_audio_output_device)
                sd.wait()
                radio.ptt_off()
    finally:
        radio.close()

if __name__ == '__main__':
    main()
