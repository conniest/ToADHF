# ggwave_terminal.py
from radio_common import IC7300, K3S, FTDX10
from ggwave_decoder import decode_wav_file, decode_wav_file_multi

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
import threading, time, os

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

usb_audio_output_device = "USB Audio CODEC"

import numpy as np



def detect_sfd_pair_or_fallback(data, fs, band=(350, 3500),
                                baseline_window_sec=2.0,
                                min_duration_sec=0.1,
                                min_rise_db=10,
                                min_sfd_duration_sec=0.22,
                                min_bandwidth_hz=700,
                                fallback_duration=1.0,
                                pad_sec=0.15):
    """
    Detect GGWave start/end frame delimiters or fall back to a fixed window if only one is found.
    Returns (start_sample, end_sample) or (None, None).
    """
    from scipy.signal import spectrogram

    nperseg = 1024
    noverlap = 512
    f, t, Sxx = spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)

    f_mask = (f >= band[0]) & (f <= band[1])
    band_power = Sxx[f_mask].mean(axis=0)
    band_power_db = 10 * np.log10(band_power + 1e-10)

    baseline_win = int((baseline_window_sec * fs) / (nperseg - noverlap))
    smoothed_baseline = np.convolve(band_power_db, np.ones(baseline_win) / baseline_win, mode='same')
    delta_db = band_power_db - smoothed_baseline

    min_frames = int((min_duration_sec / (nperseg / fs)) + 0.5)
    sfd_regions = []
    count = 0

    for i, val in enumerate(delta_db > min_rise_db):
        if val:
            count += 1
            if count == min_frames:
                start_idx = i - count + 1
                region_start = t[start_idx]
        else:
            if count >= min_frames:
                end_idx = i
                region_end = t[end_idx]
                duration = region_end - region_start

                spec_slice = Sxx[:, start_idx:end_idx]
                avg_spectrum = spec_slice.mean(axis=1)
                power_db = 10 * np.log10(avg_spectrum + 1e-10)
                active_band = f[power_db > power_db.max() - 10]
                bandwidth = active_band[-1] - active_band[0] if len(active_band) > 1 else 0

                if duration >= min_sfd_duration_sec and bandwidth >= min_bandwidth_hz:
                    sfd_regions.append((region_start, region_end))
            count = 0

    if count >= min_frames:
        end_idx = len(t) - 1
        region_end = t[end_idx]
        duration = region_end - region_start
        spec_slice = Sxx[:, start_idx:]
        avg_spectrum = spec_slice.mean(axis=1)
        power_db = 10 * np.log10(avg_spectrum + 1e-10)
        active_band = f[power_db > power_db.max() - 10]
        bandwidth = active_band[-1] - active_band[0] if len(active_band) > 1 else 0
        if duration >= min_sfd_duration_sec and bandwidth >= min_bandwidth_hz:
            sfd_regions.append((region_start, region_end))

    if len(sfd_regions) >= 2:
        s1_start, _ = sfd_regions[0]
        _, s2_end = sfd_regions[1]
        start_sample = max(0, int((s1_start - pad_sec) * fs))
        end_sample = min(len(data), int((s2_end + pad_sec) * fs))
        return start_sample, end_sample

    if len(sfd_regions) == 1:
        s1_start, _ = sfd_regions[0]
        start_sample = max(0, int((s1_start - pad_sec) * fs))
        end_sample = min(len(data), int((s1_start + fallback_duration + pad_sec) * fs))
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


def listen_loop(session, radio, device="USB Audio CODEC", samplerate=48000):
    record_duration = 10.0
    output_dir = "recordings"
    os.makedirs(output_dir, exist_ok=True)

    with patch_stdout():
        print(f"[ToAD] Listening on {radio} ('{device}')")
        stream = sd.InputStream(device=device, channels=1, samplerate=samplerate, dtype='float32')
        stream.start()

        while True:
            audio, _ = stream.read(int(record_duration * samplerate))
            filename = os.path.join(output_dir, f"toad_{time.strftime('%Y%m%d_%H%M%S')}.wav")
            sf.write(filename, audio, samplerate)

            try:
                results = decode_wav_file_multi(filename)
            except RuntimeError:
                print(f"[ToAD] No decode from last {record_duration}s")
                continue

            for msg in results:
                print(f"[RECV] {msg}")

def main():
    radio = FTDX10()
    radio.set_mode('DATA')
    samplerate = 48000

    # Build our PromptSession once
    session = PromptSession("[SEND] > ")

    # Start listener thread, passing the session so it can print above it
    rx_thread = threading.Thread(target=listen_loop, args=(session, radio), daemon=True)
    rx_thread.start()
    time.sleep(0.5)

    try:
        while True:
            # prompt_toolkit will redraw the [SEND] > prompt after any print()
            text = session.prompt().upper()
            multiplied_text = ""
            for char in text:
                multiplied_text += char*4
            payload = ggwave.encode(multiplied_text, protocolId=1)

            with radio.tx_lock:
                radio.ptt_on()
                samples = np.frombuffer(payload, dtype=np.float32)
                padding = np.zeros(int(0.02 * samplerate), dtype=np.float32)
                sd.play(np.concatenate([padding, samples, padding]), samplerate=samplerate,
                        device=usb_audio_output_device)
                sd.wait()
                radio.ptt_off()

    finally:
        radio.close()

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
