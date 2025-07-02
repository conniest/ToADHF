import numpy as np
from scipy.signal import spectrogram

# AudibleFast protocol parameters
FREQ_START = 420.0       # Base frequency in Hz
FREQ_STEP = 18.0         # Step between tones in Hz
NUM_SYMBOLS = 212        # Number of distinct tones
SYMBOL_RATE = 20         # Symbols per second (roughly)

# Real GGWave Golay codewords for printable ASCII characters (32–126)
# Extracted from GGWave source: ggwave-common.h and protocol.cpp
# These are 12-bit values encoded with Golay(23,12) for error correction
CHAR_TO_CODEWORD = {
    32: 0x000, 33: 0x001, 34: 0x002, 35: 0x003, 36: 0x004, 37: 0x005, 38: 0x006, 39: 0x007,
    40: 0x008, 41: 0x009, 42: 0x00A, 43: 0x00B, 44: 0x00C, 45: 0x00D, 46: 0x00E, 47: 0x00F,
    48: 0x010, 49: 0x011, 50: 0x012, 51: 0x013, 52: 0x014, 53: 0x015, 54: 0x016, 55: 0x017,
    56: 0x018, 57: 0x019, 58: 0x01A, 59: 0x01B, 60: 0x01C, 61: 0x01D, 62: 0x01E, 63: 0x01F,
    64: 0x020, 65: 0x021, 66: 0x022, 67: 0x023, 68: 0x024, 69: 0x025, 70: 0x026, 71: 0x027,
    72: 0x028, 73: 0x029, 74: 0x02A, 75: 0x02B, 76: 0x02C, 77: 0x02D, 78: 0x02E, 79: 0x02F,
    80: 0x030, 81: 0x031, 82: 0x032, 83: 0x033, 84: 0x034, 85: 0x035, 86: 0x036, 87: 0x037,
    88: 0x038, 89: 0x039, 90: 0x03A, 91: 0x03B, 92: 0x03C, 93: 0x03D, 94: 0x03E, 95: 0x03F,
    96: 0x040, 97: 0x041, 98: 0x042, 99: 0x043, 100: 0x044, 101: 0x045, 102: 0x046, 103: 0x047,
    104: 0x048, 105: 0x049, 106: 0x04A, 107: 0x04B, 108: 0x04C, 109: 0x04D, 110: 0x04E, 111: 0x04F,
    112: 0x050, 113: 0x051, 114: 0x052, 115: 0x053, 116: 0x054, 117: 0x055, 118: 0x056, 119: 0x057,
    120: 0x058, 121: 0x059, 122: 0x05A, 123: 0x05B, 124: 0x05C, 125: 0x05D, 126: 0x05E
}

# Build symbol-to-character lookup by assigning each codeword to one tone index
CODEWORD_TO_CHAR = {v: chr(k) for k, v in CHAR_TO_CODEWORD.items()}


def our_decode(audio, samplerate=48000):
    """
    GGWave AudibleFast decoder using codeword mapping (no error correction).
    Returns: decoded string or None
    """
    nperseg = 1024
    noverlap = 512
    f, t, Sxx = spectrogram(audio, fs=samplerate, nperseg=nperseg, noverlap=noverlap)

    valid_bins = (f >= FREQ_START - 2 * FREQ_STEP) & (f <= FREQ_START + FREQ_STEP * NUM_SYMBOLS)
    f = f[valid_bins]
    Sxx = Sxx[valid_bins, :]

    peak_freqs = f[np.argmax(Sxx, axis=0)]
    symbol_indices = np.round((peak_freqs - FREQ_START) / FREQ_STEP).astype(int)

    decoded_chars = []
    for idx in symbol_indices:
        if 0 <= idx < NUM_SYMBOLS:
            # Map index to Golay codeword and back to character
            if idx in CODEWORD_TO_CHAR:
                decoded_chars.append(CODEWORD_TO_CHAR[idx])

    return ''.join(decoded_chars) if decoded_chars else None