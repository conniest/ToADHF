import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import sys
from collections import Counter

from config import *

# === Codebook Definitions ===
from ggwave_alphabet import TEXT_TO_GGWAVE, GGWAVE_TO_TEXT

# === Spectrogram & I/O ===
def compute_spectrogram(waveform, rate):
    _, _, spec = scipy.signal.stft(
        waveform, fs=rate, nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP_SIZE, window=WINDOW,
        padded=False, boundary=None)
    return np.abs(spec)


def load_wav(filename):
    rate, data = wav.read(filename)
    if data.ndim > 1:
        data = data[:, 0]
    return rate, data.astype(np.float32)

# === Boundary Detection ===
# === Boundary Detection ===
def detect_start_frame(spectrogram, freqs):
    """
    Find the first run of at least 3 consecutive high-energy, all-ones frames.
    Returns (start_idx, end_idx, energy, threshold, min_idx, max_idx).
    """
    # locate frequency bin range for GGWave tones
    min_idx = np.argmin(np.abs(freqs - GGWAVE_FREQ_MIN))
    max_idx = np.argmin(np.abs(freqs - (GGWAVE_FREQ_MIN + GGWAVE_FREQ_STEP * (GGWAVE_NUM_TONES - 1))))
    energy = np.sum(spectrogram[min_idx:max_idx+1, :], axis=0)
    threshold = 0.5 * np.mean(energy)

    # precompute tone-bin indices
    tone_bins = [
        np.argmin(np.abs(freqs - (GGWAVE_FREQ_MIN + j * GGWAVE_FREQ_STEP)))
        for j in range(GGWAVE_NUM_TONES)
    ]
    all_ones = np.ones(GGWAVE_NUM_TONES, dtype=np.uint8)

    def frame_bits(col):
        # extract energy per tone at column `col`
        en = np.array([
            np.sum(spectrogram[max(0, b-1):min(b+2, spectrogram.shape[0]), col])
            for b in tone_bins
        ])
        return (en > 0.5 * en.max()).astype(np.uint8)

    # scan for 3+ consecutive high‐energy frames whose first two (or three) are all‐ones
    for i in range(len(energy) - 2):
        if energy[i] > threshold and energy[i+1] > threshold and energy[i+2] > threshold:
            # require the first 3 frames to have exactly the all‐ones pattern
            if (np.array_equal(frame_bits(i), all_ones)
                and np.array_equal(frame_bits(i+1), all_ones)
                and np.array_equal(frame_bits(i+2), all_ones)):
                # now extend the run until energy drops below threshold
                j = i + 3
                while np.array_equal(frame_bits(j), all_ones):
                    j += 1
                end_idx = j - 1
                # print(f"[INFO] SFD starts at column {i}, ends at column {end_idx}")
                return i, end_idx, energy, threshold, min_idx, max_idx

    raise RuntimeError(
        "No valid SFD found: need ≥3 consecutive high‐energy, all-ones frames"
    )



def detect_end_frame(spectrogram, freqs, signal_start=0):
    """
    Find first column where at least 3 consecutive frames exceed threshold
    AND the first two frames have an all-ones tone pattern.
    Returns (index, energy, threshold, min_idx, max_idx).
    """
    # locate frequency bin range for GGWave tones
    min_idx = np.argmin(np.abs(freqs - GGWAVE_FREQ_MIN))
    max_idx = np.argmin(np.abs(freqs - (GGWAVE_FREQ_MIN + GGWAVE_FREQ_STEP * (GGWAVE_NUM_TONES - 1))))
    energy = np.sum(spectrogram[min_idx:max_idx+1, :], axis=0)
    threshold = 0.5 * np.mean(energy)
    # precompute exact tone-bin indices
    tone_bins = [np.argmin(np.abs(freqs - (GGWAVE_FREQ_MIN + j * GGWAVE_FREQ_STEP)))
                 for j in range(GGWAVE_NUM_TONES)]
    # helper to compute bit pattern for a given STFT column
    def frame_bits(col):
        energies = np.array([
            np.sum(spectrogram[max(0, b-1):min(b+2, spectrogram.shape[0]), col])
            for b in tone_bins
        ])
        max_e = np.max(energies)
        return (energies > 0.5 * max_e).astype(np.uint8)

    all_ones = np.ones(GGWAVE_NUM_TONES, dtype=np.uint8)
    # scan for 3 high-energy in a row
    for i in range(signal_start, len(energy) - 2):
        if (energy[i] > threshold and energy[i+1] > threshold and energy[i+2] > threshold):
            # require exact all-ones pattern in first two frames
            bits0 = frame_bits(i)
            bits1 = frame_bits(i+1)
            bits2 = frame_bits(i+2)
            if np.array_equal(bits0, all_ones) and np.array_equal(bits1, all_ones) and np.array_equal(bits2, all_ones):
                #print(f"[INFO] EFD starts at column {i}")
                # now extend the run until energy drops below threshold
                j = i + 3
                while np.array_equal(frame_bits(j), all_ones):
                    j += 1
                end_idx = j - 1
                return i, end_idx
    raise RuntimeError("No valid EFD found (need 3 consecutive high-energy frames with first two all-ones)")

# === Binarization & Trimming ===
def binarize_symbol_frames(symbol_frames, threshold_ratio=0.5):
    bits = []
    for frame in symbol_frames:
        max_e = np.max(frame)
        bits.append((frame > (threshold_ratio * max_e)).astype(np.uint8))
    return np.array(bits)


def trim_and_binarize_between(spectrogram, freqs, metadata_frames=11, threshold_ratio=0.5):
    start, start_e, energy, thresh, mi, ma = detect_start_frame(spectrogram, freqs)
    end, end_e = detect_end_frame(spectrogram, freqs, signal_start=start_e+metadata_frames)
    idxs = list(range(start, end+1, GGWAVE_SYMBOL_HOP_FRAMES))
    tone_bins = [np.argmin(np.abs(freqs - (GGWAVE_FREQ_MIN + i * GGWAVE_FREQ_STEP)))
                 for i in range(GGWAVE_NUM_TONES)]
    frames = []
    for t in idxs:
        frames.append(np.array([
            np.sum(spectrogram[max(0, b-1):min(b+2, spectrogram.shape[0]), t])
            for b in tone_bins]))
    bits_all = binarize_symbol_frames(np.array(frames), threshold_ratio)
    sfd_pat = np.ones(GGWAVE_NUM_TONES, dtype=np.uint8)
    is_data = np.any(bits_all != sfd_pat, axis=1)
    first_d = np.argmax(is_data)
    last_d = len(is_data)-1 - np.argmax(is_data[::-1])
    start_p = first_d + metadata_frames
    end_p = last_d - metadata_frames
    return bits_all[start_p:end_p]

# === Decoding Helpers ===
def decode_payload_runs(bits):
    bitstrs = [''.join(str(x) for x in frame) for frame in bits]
    chars = [GGWAVE_TO_TEXT.get(bs, '?') for bs in bitstrs]
    runs, prev, count = [], chars[0], 1
    for c in chars[1:]:
        if c == prev:
            count += 1
        else:
            runs.append(count)
            prev, count = c, 1
    runs.append(count)
    return runs


def estimate_frames_per_char(runs):
    return int(np.median(runs))


# === Fuzzy Decode ===
def decode_message_fuzzy(bits, frames_per_char=16):
    """
    Perform a majority-vote across each block of `frames_per_char` frames.
    If one symbol wins >50%, pick it. If a 2-2 tie, return both as "[a|b]".
    """
    bitstrs = [''.join(str(x) for x in frame) for frame in bits]
    total = len(bitstrs)
    char_count = total // frames_per_char
    decoded = []
    for i in range(char_count):
        slice_ = bitstrs[i*frames_per_char:(i+1)*frames_per_char]
        # map each frame to char
        chars = [GGWAVE_TO_TEXT.get(bs, '?') for bs in slice_]
        # count votes
        votes = {}
        for c in chars:
            votes[c] = votes.get(c, 0) + 1
        # find winners
        max_votes = max(votes.values())
        winners = [c for c, v in votes.items() if v == max_votes]
        if len(winners) == 1:
            decoded.append(winners[0])
        else:
            # tie -> list both
            decoded.append("[" + "|".join(sorted(winners)) + "]")
    return ''.join(decoded)


from collections import Counter

def decode_redundant_message(decoded_chars, redundancy=4, frames_per_char=4):
    """
    Take a sequence of fuzzy-decoded symbols (length ≥ redundancy), in which each
    "real" symbol was sent redundancy times.  Split into blocks of size redundancy,
    then for each block:

      • If a token is a single char (e.g. 'K'), it contributes frames_per_char votes
        to that char.
      • If a token is a tie like "[A|B]", it contributes frames_per_char/2 votes to A
        and frames_per_char/2 votes to B.
      • (You can extend to "[A|B|C]" by splitting on '|' and dividing equally.)

    After summing all votes in the block, pick whichever char(s) have the highest
    total.  If there’s still a multi-way tie at the end, emit "[A|B|…]".  Any
    leftover tail chars (len(decoded_chars) % redundancy) are dropped.

    Returns a single string of length floor(len(decoded_chars)/redundancy).
    """

    final = []
    length = len(decoded_chars)
    usable = (length // redundancy) * redundancy

    for start in range(0, usable, redundancy):
        block = decoded_chars[start : start + redundancy]

        # Accumulate weighted votes
        scores = {}
        for token in block:
            if token.startswith('[') and token.endswith(']'):
                # tie-token → multiple candidates
                candidates = token[1:-1].split('|')
            else:
                candidates = [token]

            weight = frames_per_char / len(candidates)
            for c in candidates:
                scores[c] = scores.get(c, 0) + weight

        # Pick the winner(s)
        max_score = max(scores.values())
        winners = sorted(c for c, v in scores.items() if v == max_score)

        if len(winners) == 1:
            final.append(winners[0])
        else:
            final.append(f"[{'|'.join(winners)}]")

    return ''.join(final)


# === Utility: Frame/Sample Calculators ===
def calc_hop_size_for_target_frames(target_frames, sample_rate=GGWAVE_SAMPLE_RATE, symbol_rate=GGWAVE_SYMBOL_RATE):
    return int(sample_rate / (symbol_rate * target_frames))


def calc_sample_rate_for_target_frames(target_frames, hop_size=HOP_SIZE, symbol_rate=GGWAVE_SYMBOL_RATE):
    return int(hop_size * symbol_rate * target_frames)

# === Programmatic interface ===
def decode_wav_file(filename, frames_per_char=4):
    """
    Load a WAV file and run the full GGWave fuzzy decoder pipeline,
    returning the decoded message string.
    """
    # load audio
    rate, waveform = load_wav(filename)
    if rate != GGWAVE_SAMPLE_RATE:
        print(f"Warning: sample rate {rate} != expected {GGWAVE_SAMPLE_RATE}")
    # compute spectrogram
    spec = compute_spectrogram(waveform, rate)
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0/rate)
    # trim and binarize payload
    bits = trim_and_binarize_between(spec, freqs)
    for i, b in enumerate(bits):
        print(f"[DEBUG] {i}: {''.join(str(x) for x in b)}")
    # fuzzy-decode message
    msg = decode_message_fuzzy(bits, frames_per_char)
    final_msg = decode_redundant_message(msg)
    if msg:
        return final_msg
    return None

# === Multi-Transmission Decode API ===
def decode_wav_file_multi(filename, frames_per_char=4):
    """
    Decode all GGWave bursts in <filename>, returning a list of decoded strings.
    Uses trim_and_binarize_between + decode_message_fuzzy under the hood,
    and then masks out each burst before looking for the next.
    """
    rate, waveform = load_wav(filename)
    # if rate != GGWAVE_SAMPLE_RATE:
    #    print(f"Warning: sample rate {rate} != expected {GGWAVE_SAMPLE_RATE}")
    spec = compute_spectrogram(waveform, rate)
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0/rate)

    spec_work = spec.copy()
    messages = []

    while True:
        # Try to trim & binarize the *next* burst
        try:
            bits = trim_and_binarize_between(spec_work, freqs)
            #for i, b in enumerate(bits):
            #    print(f"[DEBUG] {i}: {''.join(str(x) for x in b)}")
        except RuntimeError:
            break  # no more SFD/EFD pairs → done

        # Fuzzy-decode that burst
        msg = decode_message_fuzzy(bits, frames_per_char)
        final_msg = decode_redundant_message(msg)
        messages.append(final_msg)

        # Now locate and zero out exactly that burst in the spectrogram
        # so the next iteration finds the following one.
        s_start, s_end, energy, threshold, min_idx, max_idx = detect_start_frame(spec_work, freqs)
        e_start, e_end = detect_end_frame(spec_work, freqs, signal_start=s_end)
        spec_work[:, s_start:e_end+1] = 0  # mask out columns of that burst

    return messages



# === Main ===
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.wav>")
        sys.exit(1)

    target = 4
    hop_needed = calc_hop_size_for_target_frames(target)
    rate_needed = calc_sample_rate_for_target_frames(target)
    #print(f"For ~{target} frames/symbol at {GGWAVE_SAMPLE_RATE}Hz, HOP_SIZE≈{hop_needed}")
    #print(f"Or keep HOP_SIZE={HOP_SIZE} and use sample rate≈{rate_needed}Hz")

    rate, wavf = load_wav(sys.argv[1])
    if rate != GGWAVE_SAMPLE_RATE:
        print(f"Warning: input rate {rate} != expected {GGWAVE_SAMPLE_RATE}")
    spec = compute_spectrogram(wavf, rate)
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0/rate)

    bits = trim_and_binarize_between(spec, freqs)
    for i, b in enumerate(bits):
        print(f"[DEBUG] {i}: {''.join(str(x) for x in b)}")

    #runs = decode_payload_runs(bits)
    fpc = 4
    msg = decode_message_fuzzy(bits, fpc)
    print(f"Frames/char: {fpc}")
    print(f"Decoded: {msg}")

if __name__ == '__main__':
    main()
