# ggwave_terminal.py
from radio_common import IC7300, K3S
from ggwave_decoder import our_decode

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

from ggwave_alphabet import GGWAVE_CODEBOOK

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


def listen_loop(radio, device="USB Audio CODEC", samplerate=48000):
    record_duration = 5.0  # seconds per chunk
    output_dir = "/home/glick/Desktop/CATpack/src/recordings/chars"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[ToAD] Listening on '{device}' – saving {record_duration}s chunks to disk")

    frames_per_chunk = int(record_duration * samplerate)
    stream = sd.InputStream(device=device, channels=1, samplerate=samplerate, dtype='float32')
    stream.start()

    

    for ch in GGWAVE_CODEBOOK.keys():
        # if radio.tx_lock.locked():
        #     time.sleep(0.1)
        #     continue

        print("[ToAD] Capturing audio chunk...")
        audio, _ = stream.read(frames_per_chunk)

        filename = os.path.join(
            output_dir,
            f"toad_{ch}.wav"
        )
        sf.write(filename, audio, samplerate)
        print(f"[ToAD] Saved → {filename}")

        pcm = audio[:, 0].astype(np.float32)  # mono

        # Use pcm directly for detection to keep indices aligned
        start, end = detect_sfd_pair_or_fallback(pcm, samplerate)
        result = None

        if start is not None and end is not None:
            print(f"[ToAD] Detected GGWave burst from {start/samplerate:.2f}s to {end/samplerate:.2f}s")

            clip = pcm[start:end]

            # Normalize to peak = 0.9
            max_val = np.max(np.abs(clip))
            if max_val > 0:
                clip = clip / max_val * 0.9

            print(f"[DEBUG] start={start}, end={end}, len={end - start}")
            print(f"[DEBUG] dtype={clip.dtype}, max={np.max(np.abs(clip)):.4f}")
            print(f"[DEBUG] bytes={len(clip.astype(np.float32).tobytes())}")
            #result = our_decode(clip.astype(np.float32))
            params = ggwave.getDefaultParameters()
            print(params)
            params["sampleRateInp"] = 48000.0     # Confirm this is your USB CODEC rate
            params["sampleRate"] = 48000.0        # Keep consistent
            params["soundMarkerThreshold"] = 2.5  # Try lowering if burst isn't detected reliably
            ctx = ggwave.init(params)
            result = ggwave.decode(ctx, clip.astype(np.float32).tobytes())
        else:
            print(f"[ToAD] No GGWave SFD detected in {filename}")

        if result:
            print(f"\n[RECV] {result}\n> ", end='', flush=True)
        else:
            print(f"[ToAD] No decode from {filename}")
def main():
    radio = IC7300()
    radio.set_mode('LSB-D')
    samplerate = 48000

    # how much silence to pad before/after each burst
    pre_silence  = 0.5  # seconds
    post_silence = 0.5  # seconds

    output_dir = "recordings/chars"
    os.makedirs(output_dir, exist_ok=True)

    try:
        for ch in GGWAVE_CODEBOOK.keys():
            print(f"[ToAD] Preparing character '{ch}'")

            # --- build the GGWave payload ---
            text = ch * 10
            payload = ggwave.encode(text.encode(), protocolId=1)
            samples = np.frombuffer(payload, dtype=np.float32)

            # pad it so we always record even with latency
            pre  = np.zeros(int(pre_silence  * samplerate), dtype=np.float32)
            post = np.zeros(int(post_silence * samplerate), dtype=np.float32)
            outbuf = np.concatenate([pre, samples, post]).reshape(-1, 1)

            # --- open a full-duplex stream on the USB codec ---
            with sd.Stream(device=(usb_audio_output_device, usb_audio_output_device),
                           samplerate=samplerate,
                           channels=1,
                           dtype='float32',
                           latency='low') as stream:

                # key the transmitter *just* around the burst
                radio.ptt_on()
                # write() will block until outbuf has been sent *and* inbuf filled
                inbuf, _ = stream.write(outbuf)  
                radio.ptt_off()

            pcm = inbuf[:, 0]  # collapse to 1-d array

            # --- save the raw capture ---
            filename = os.path.join(output_dir, f"toad_{ch}.wav")
            sf.write(filename, pcm, samplerate)
            print(f"[ToAD] Saved → {filename}")

            # --- detect & decode exactly where the burst landed ---
            start, end = detect_sfd_pair_or_fallback(pcm, samplerate)
            if start is not None and end is not None:
                print(f"[ToAD] Detected GGWave burst from {start/samplerate:.2f}s to {end/samplerate:.2f}s")
                clip = pcm[start:end]
                clip = clip / np.max(np.abs(clip)) * 0.9

                params = ggwave.getDefaultParameters()
                params["sampleRateInp"] = samplerate
                params["sampleRate"]    = samplerate
                ctx = ggwave.init(params)
                result = ggwave.decode(ctx, clip.astype(np.float32).tobytes())
                if result:
                    print(f"\n[RECV] {result}\n")
                else:
                    print("[ToAD] GGWave decode failed")
            else:
                print("[ToAD] No GGWave burst detected")

            time.sleep(2.0)

    finally:
        radio.close()




if __name__ == '__main__':
    main()
