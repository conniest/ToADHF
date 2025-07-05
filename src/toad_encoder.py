import numpy as np
import scipy.io.wavfile as wav
from ggwave_alphabet import TEXT_TO_GGWAVE

from config import *

# === GGWave Configurations ===
TOAD_SAMPLE_RATE = 48_000          # Hz
TOAD_SYMBOL_RATE = 8               # 1 symbol = 1/8 s → 8 symbols/s
TOAD_NUM_TONES   = 16
TOAD_FREQ_MIN    = 100.0          # Hz of the first bin
TOAD_FREQ_STEP   = 175         # Hz spacing between bins

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _raised_cosine(fade_samps: int) -> np.ndarray:
    """Half‑cosine window from 0→1 (len = fade_samps)."""
    return 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samps)))


def _smooth_gate(bit_vec: np.ndarray, fade_samps: int) -> np.ndarray:
    """Return a smoothed 0/1 gate (raised‑cosine on/off)."""
    if fade_samps == 0:
        return bit_vec.astype(np.float32)

    win = _raised_cosine(fade_samps)
    # build symmetric window 0→1→0 (len = 2*fade)
    win_full = np.concatenate([win, win[::-1]])
    # normalise so plateau stays at 1 when consecutive symbols are 1‑1‑1 …
    win_full /= win_full.max()
    # Convolve & trim
    g = np.convolve(bit_vec.astype(np.float32), win_full, mode="same")
    return np.clip(g, 0.0, 1.0)

# -----------------------------------------------------------------------------
# Encoder (continuous‑phase, per‑tone raised‑cosine smoothing, power EQ)
# -----------------------------------------------------------------------------

def encode_text_to_waveform(text: str,
                            preamble_len: int = 8,
                            amplitude: float = 0.8,
                            fade_ms: float = 4.0) -> np.ndarray:
    """Return GGWave‑compatible float32 waveform in [‑1,1]."""

    sr        = TOAD_SAMPLE_RATE
    sym_dur   = 1.0 / TOAD_SYMBOL_RATE
    sym_samps = int(round(sr * sym_dur))
    fade_samp = min(sym_samps // 2, int(round(sr * fade_ms / 1000)))

    # Build symbol patterns ----------------------------------------------------
    all_ones  = "1" * TOAD_NUM_TONES
    patterns  = [all_ones] * preamble_len + \
                [TEXT_TO_GGWAVE.get(ch, all_ones) for ch in text] + \
                [all_ones] * preamble_len
    n_syms    = len(patterns)

    # Binary matrix (syms × tones) --------------------------------------------
    bit_mat   = np.array([[int(b) for b in pat] for pat in patterns], dtype=np.uint8)

    # For each tone, build long 0/1 vector and smooth -------------------------
    total_samps  = n_syms * sym_samps
    time_index   = np.arange(total_samps, dtype=np.float32) / sr

    freqs = TOAD_FREQ_MIN + np.arange(TOAD_NUM_TONES) * TOAD_FREQ_STEP
    two_pi = 2 * np.pi

    waveform = np.zeros(total_samps, dtype=np.float32)
    active_cnt = np.zeros(total_samps, dtype=np.float32)  # how many tones on per sample

    for k, freq in enumerate(freqs):
        # gate for this tone over the whole file
        gate_sym = bit_mat[:, k]                                    # (n_syms,)
        gate_long = np.repeat(gate_sym, sym_samps)                  # (total_samps,)
        gate_long = _smooth_gate(gate_long, fade_samp)              # raised‑cosine edges

        # continuous phase -----------------------------------------------------
        phase_inc = two_pi * freq / sr
        phase = phase_inc * np.arange(total_samps, dtype=np.float32)
        tone = np.sin(phase)

        waveform += tone * gate_long
        active_cnt += gate_long

    # Power equalisation: scale by 1 / max(1, n_active) -----------------------
    active_cnt = np.maximum(active_cnt, 1.0)
    waveform  /= active_cnt

    # Normalise final amplitude -----------------------------------------------
    waveform *= amplitude / np.max(np.abs(waveform))
    return waveform.astype(np.float32)


# -----------------------------------------------------------------------------
# Convenience I/O
# -----------------------------------------------------------------------------

def write_waveform_to_wav(waveform: np.ndarray, filename: str) -> None:
    wav.write(filename, TOAD_SAMPLE_RATE, waveform)


# -----------------------------------------------------------------------------
# Quick CLI test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    plain_msg = "HELLO WORLD"
    re_msg = "".join([c * 4 for c in plain_msg])
    wf  = encode_text_to_waveform(re_msg)
    write_waveform_to_wav(wf, "ggwave_encoded.wav")
    print("Saved encoded GGWave to ggwave_encoded.wav")
