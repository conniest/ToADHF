import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import sys

# === GGWave AudibleFast Parameters ===
GGWAVE_SAMPLE_RATE = 48000
GGWAVE_SYMBOL_RATE = 30
GGWAVE_NUM_TONES = 25
GGWAVE_FREQ_MIN = 1875.0
GGWAVE_FREQ_STEP = 46.875
GGWAVE_SYMBOL_DURATION_S = 1.0 / GGWAVE_SYMBOL_RATE
GGWAVE_SYMBOL_HOP_FRAMES = 6  # ~30 Hz update rate in STFT

# === FFT Parameters ===
FFT_SIZE = 1024
HOP_SIZE = FFT_SIZE // 4  # 75% overlap
WINDOW = np.hanning(FFT_SIZE)


def compute_spectrogram(waveform, rate):
    _, _, spec = scipy.signal.stft(
        waveform,
        fs=rate,
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP_SIZE,
        window=WINDOW,
        padded=False,
        boundary=None,
    )
    return np.abs(spec)


def load_wav(filename):
    rate, data = wav.read(filename)
    if data.ndim > 1:
        data = data[:, 0]
    return rate, data.astype(np.float32)


def detect_start_frame(spectrogram, freqs):
    min_idx = np.argmin(np.abs(freqs - GGWAVE_FREQ_MIN))
    max_idx = np.argmin(
        np.abs(freqs - (GGWAVE_FREQ_MIN + GGWAVE_FREQ_STEP * (GGWAVE_NUM_TONES - 1)))
    )
    energy = np.sum(spectrogram[min_idx:max_idx + 1, :], axis=0)
    threshold = 0.5 * np.mean(energy)
    for i, e in enumerate(energy):
        if e > threshold:
            print(f"[INFO] Auto-detected SFD at spectrogram column {i}")
            return i, energy, threshold, min_idx, max_idx
    raise RuntimeError("Failed to detect valid start of symbol sequence")


def detect_end_frame(spectrogram, freqs, energy, threshold, min_idx, max_idx):
    for i in range(len(energy) - 1, -1, -1):
        if energy[i] > threshold:
            print(f"[INFO] Auto-detected EFD at spectrogram column {i}")
            return i
    raise RuntimeError("Failed to detect valid end of symbol sequence")


def binarize_symbol_frames(symbol_frames, threshold_ratio=0.5):
    bits = []
    for frame in symbol_frames:
        max_energy = np.max(frame)
        bits.append((frame > (threshold_ratio * max_energy)).astype(np.uint8))
    return np.array(bits)


def trim_and_binarize_between(spectrogram, freqs, metadata_frames=4, threshold_ratio=0.5):
    """
    Locate SFD/EFD via energy, collect all symbol frames between them,
    binarize them, then trim SFD/preamble and EFD/postamble bit frames.
    metadata_frames: number of frames of metadata after start and before end
    """
    # Detect start and end delimiters (energy-based)
    start_idx, energy, threshold, min_idx, max_idx = detect_start_frame(spectrogram, freqs)
    end_idx = detect_end_frame(spectrogram, freqs, energy, threshold, min_idx, max_idx)
    print(f"[INFO] Symbol block from column {start_idx} to {end_idx}")

    # Build symbol frame indices
    frame_indices = list(range(start_idx, end_idx + 1, GGWAVE_SYMBOL_HOP_FRAMES))
    if not frame_indices:
        raise RuntimeError("No symbol frames detected between start and end delimiters")

    # Extract energy-based symbol frames for all bins
    tone_bins = [np.argmin(np.abs(freqs - (GGWAVE_FREQ_MIN + i * GGWAVE_FREQ_STEP)))
                 for i in range(GGWAVE_NUM_TONES)]
    symbol_frames = []
    for t in frame_indices:
        energies = np.array([
            np.sum(spectrogram[max(0, b - 1):min(b + 2, spectrogram.shape[0]), t])
            for b in tone_bins
        ])
        symbol_frames.append(energies)
    symbol_frames = np.array(symbol_frames)

    # Binarize all frames
    bits_all = binarize_symbol_frames(symbol_frames, threshold_ratio=threshold_ratio)

    # Identify frames that differ from SFD pattern (all 1s)
    sfd_pattern = np.ones(GGWAVE_NUM_TONES, dtype=np.uint8)
    is_data = np.any(bits_all != sfd_pattern, axis=1)
    if not np.any(is_data):
        raise RuntimeError("Could not find payload frames between delimiters")
    first_data = np.argmax(is_data)
    last_data = len(is_data) - 1 - np.argmax(is_data[::-1])

    # Advance past metadata at start and back before metadata at end
    data_start = first_data + metadata_frames
    data_end = last_data - metadata_frames
    if data_start > data_end:
        raise RuntimeError("Metadata trimming removed all payload frames")

    # Slice out payload bits
    payload_bits = bits_all[data_start:data_end + 1]
    return payload_bits


def dump_bits(bits):
    for i, b in enumerate(bits):
        print(f"[DEBUG] Symbol frame {i}: bits = {''.join(str(x) for x in b)}")

# Decoding helpers (bits_to_int, hamming_distance, decode_audiblefast_bitstrings) unchanged


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.wav>")
        sys.exit(1)

    wav_path = sys.argv[1]
    rate, waveform = load_wav(wav_path)
    if rate != GGWAVE_SAMPLE_RATE:
        raise ValueError(f"Expected sample rate {GGWAVE_SAMPLE_RATE}, got {rate}")

    spectrogram = compute_spectrogram(waveform, rate)
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / rate)
    print(f"[INFO] Spectrogram shape: freq bins = {spectrogram.shape[0]}, time steps = {spectrogram.shape[1]}")

    bits = trim_and_binarize_between(spectrogram, freqs)
    dump_bits(bits)

    # TODO: Decode bits with Reed-Solomon and GGWave codebook

if __name__ == "__main__":
    main()
