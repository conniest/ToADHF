import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import sys

# === GGWave AudibleFast Parameters ===
GGWAVE_SAMPLE_RATE = 48000
GGWAVE_SYMBOL_RATE = 30
GGWAVE_NUM_TONES = 64
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
        data = data[:, 0]  # mono
    return rate, data.astype(np.float32)


def detect_start_frame(spectrogram, freqs):
    # Try to find a run of frames with many active tones in GGWave band
    min_idx = np.argmin(np.abs(freqs - GGWAVE_FREQ_MIN))
    max_idx = np.argmin(np.abs(freqs - (GGWAVE_FREQ_MIN + GGWAVE_FREQ_STEP * (GGWAVE_NUM_TONES - 1))))

    energy = np.sum(spectrogram[min_idx:max_idx + 1, :], axis=0)
    avg_energy = np.mean(energy)
    threshold = 0.5 * avg_energy
    for i in range(len(energy)):
        if energy[i] > threshold:
            print(f"[INFO] Auto-detected symbol frame starting at spectrogram column {i}")
            return i
    raise RuntimeError("Failed to detect valid start of symbol sequence")


def extract_symbol_frames(spectrogram, freqs, start_frame, max_symbols=64):
    tone_bins = [np.argmin(np.abs(freqs - (GGWAVE_FREQ_MIN + i * GGWAVE_FREQ_STEP))) for i in range(GGWAVE_NUM_TONES)]
    symbol_frames = []
    for sym_idx in range(max_symbols):
        t = start_frame + sym_idx * GGWAVE_SYMBOL_HOP_FRAMES
        if t >= spectrogram.shape[1]:
            break
        energies = np.array([
            np.sum(spectrogram[max(0, b - 1):min(b + 2, len(freqs)), t]) for b in tone_bins
        ])
        symbol_frames.append(energies)
    return np.array(symbol_frames)


def binarize_symbol_frames(symbol_frames, threshold_ratio=0.5):
    bits = []
    for idx, frame in enumerate(symbol_frames):
        max_energy = np.max(frame)
        bit_vector = (frame > (threshold_ratio * max_energy)).astype(np.uint8)
        bits.append(bit_vector)
    return np.array(bits)


def dump_bits(bits):
    for i, b in enumerate(bits):
        print(f"[DEBUG] Symbol frame {i}: bits = {''.join(str(x) for x in b)}")


# Converts a string of '0'/'1' to an integer
def bits_to_int(bitstr):
    return int(str(bitstr), 2)

# Computes Hamming distance between two integers interpreted as 64-bit bitfields
def hamming_distance(x, y):
    return bin(x ^ y).count("1")

# Decodes a list of bitstrings into characters using the GGWave codebook
def decode_audiblefast_bitstrings(bitstrings):
    decoded = []
    for bitstr in bitstrings:
        word = bits_to_int(bitstr)
        best_char = '?'
        min_dist = 65  # max possible distance is 64

        for i, codeword in enumerate(GGWAVE_CODEBOOK):
            dist = hamming_distance(word, codeword)
            if dist < min_dist:
                min_dist = dist
                best_char = CHARSET[i]
                if dist == 0:
                    break  # exact match found

        decoded.append(best_char)

    return ''.join(decoded)


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

    start_frame = detect_start_frame(spectrogram, freqs)
    symbol_frames = extract_symbol_frames(spectrogram, freqs, start_frame)
    bits = binarize_symbol_frames(symbol_frames)

    dump_bits(bits)
    decode_audiblefast_bitstrings(bits)

    print("[TODO] Feed bit matrix into Reed-Solomon decoder to recover payload")

if __name__ == "__main__":
    main()
