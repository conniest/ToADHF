"""Fuzzy‑robust GGWave decoder for 25‑tone / 8 symbols‑per‑second signals.

Changes in this revision (2025‑07‑05 c)
------------------------------------
* **Adaptive symbol matcher** – new `_symbols_from_bits()` turns a 25‑bit
  frame into **one or several plausible characters** based on Hamming‑distance
  ≤ `MAX_DIST` (default 4).  When several share the same minimal distance, we
  return a tie token like `[H|K]`.
* **Graceful blank detection** – frames whose active‑bit count < 3 are treated
  as `?` to avoid spurious matches.
* **Updated vote aggregator** – understands tie tokens and weights votes
  accordingly, producing a single confident character or a final tie token.
* **Slightly looser row‑binarisation** – default energy threshold ratio lowered
  to 0.45 for better robustness against weak tones.

Usage (CLI)
-----------
$ python ggwave_fuzzy_decoder.py ggwave_encoded.wav 4            # default
$ python ggwave_fuzzy_decoder.py ggwave_encoded.wav 4 --debug   # full dump
"""

from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from config import * 
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig

# -----------------------------------------------------------------------------
# Constants – must match the encoder
# -----------------------------------------------------------------------------
FFT_SIZE              = 1024
HOP_SIZE              = FFT_SIZE // 4           # 256 ⇒ 187.5 Hz update
WINDOW                = np.hanning(FFT_SIZE)

SR                    = 48_000                  # GGWave sample‑rate
SYMBOL_RATE           = 8                       # sym/s
NUM_TONES             = 16                      # 16 frequency bins
FREQ_MIN              = 100.0
FREQ_STEP             = 175.0                  # 16 × 175 = 2800 Hz span

STFT_RATE             = SR / HOP_SIZE           # ≈ 187.5 frames/s
SYMBOL_HOP_FRAMES     = int(round(STFT_RATE / SYMBOL_RATE))   # ≈ 23

MAX_DIST              = 2                       # max Hamming distance accepted
MIN_ACTIVE_BITS       = 2                       # expect 2 ones per data symbol
MARKER_MIN_BITS       = NUM_TONES - 2           # ≥14 ⇒ treat as ^ marker

# -----------------------------------------------------------------------------
# Code‑book
# -----------------------------------------------------------------------------
from ggwave_alphabet import TEXT_TO_GGWAVE, GGWAVE_TO_TEXT

_GGWAVE_BITS  = np.array([list(map(int, s)) for s in TEXT_TO_GGWAVE.values()],
                         dtype=np.uint8)
_GGWAVE_CHARS = list(TEXT_TO_GGWAVE.keys())

# -----------------------------------------------------------------------------
# STFT helpers
# -----------------------------------------------------------------------------

def _stft_mag(wave: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    freqs, _, Z = sig.stft(
        wave,
        fs=rate,
        window=WINDOW,
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP_SIZE,
        padded=False,
        boundary=None,
    )
    return np.abs(Z), freqs

# -----------------------------------------------------------------------------
# Tone bins (needs freqs)
# -----------------------------------------------------------------------------

def _tone_bins(freqs: np.ndarray) -> np.ndarray:
    return np.array([
        np.argmin(np.abs(freqs - (FREQ_MIN + i * FREQ_STEP)))
        for i in range(NUM_TONES)
    ])

# -----------------------------------------------------------------------------
# Column → 25‑bit vector
# -----------------------------------------------------------------------------

def _frame_bits(spec: np.ndarray, col: int, tone_bins: np.ndarray, thr_ratio: float = 0.45) -> np.ndarray:
    en = np.array([spec[max(0, b - 1): b + 2, col].sum() for b in tone_bins])
    return (en > thr_ratio * en.max()).astype(np.uint8)

# -----------------------------------------------------------------------------
# Frame bits → candidate symbols
# -----------------------------------------------------------------------------

def _symbols_from_bits(bits: np.ndarray) -> str:
    """Return a single char or a tie‑token like "[H|K]" or '?' for blank."""
    if bits.sum() < MIN_ACTIVE_BITS:
        return '?'
    dists = (_GGWAVE_BITS ^ bits).sum(axis=1)
    d_min = dists.min()
    if d_min > MAX_DIST:
        return '?'
    winners = [c for c, d in zip(_GGWAVE_CHARS, dists) if d == d_min]
    return winners[0] if len(winners) == 1 else f"[{'|'.join(winners)}]"

# -----------------------------------------------------------------------------
# Marker (preamble/post‑amble) detection helpers
# -----------------------------------------------------------------------------

def _majority_window(spec: np.ndarray, tone_bins: np.ndarray, *, start: int, end: int,
                     tol_bits: int, majority_windows=((10, 8), (4, 3)), min_run: int = 5,
                     reverse: bool = False) -> tuple[int, int] | None:
    cols = range(start, end) if not reverse else range(end - 1, start - 1, -1)
    target = np.ones(NUM_TONES, dtype=np.uint8)

    def good(c: int) -> bool:
        return (_frame_bits(spec, c, tone_bins) ^ target).sum() <= tol_bits

    # majority window first
    for win, need in majority_windows:
        scan = range(start, end - win + 1)
        scan = scan if not reverse else reversed(scan)
        for s in scan:
            hit = sum(good(c) for c in range(s, s + win))
            if hit >= need:
                first = next(c for c in range(s, s + win) if good(c))
                last  = max(c for c in range(first, s + win) if good(c))
                return (first, last) if not reverse else (last, first)

    # consecutive run fallback
    run_start, run_len = None, 0
    for c in cols:
        if good(c):
            run_len += 1
            if run_start is None:
                run_start = c
            if run_len >= min_run:
                return (run_start, c) if not reverse else (c, run_start)
        else:
            run_start, run_len = None, 0
    return None


def _find_marker_fwd(spec: np.ndarray, tone_bins: np.ndarray, *, search_from: int = 0, **kw) -> tuple[int, int]:
    res = _majority_window(spec, tone_bins, start=search_from, end=spec.shape[1], reverse=False, tol_bits=2, **kw)
    if res is None:
        raise RuntimeError("Marker not found (forward)")
    return res


def _find_marker_rev(spec: np.ndarray, tone_bins: np.ndarray, *, search_to: int, **kw) -> tuple[int, int]:
    res = _majority_window(spec, tone_bins, start=0, end=search_to + 1, reverse=True, tol_bits=2, **kw)
    if res is None:
        raise RuntimeError("Marker not found (reverse)")
    return res

# -----------------------------------------------------------------------------
# Redundancy vote (understands tie‑tokens)
# -----------------------------------------------------------------------------

def _vote(char_stream: list[str], redundancy: int) -> str:
    blocks = [char_stream[i:i + redundancy] for i in range(0, len(char_stream), redundancy)]
    out = []
    for blk in blocks:
        scores: Counter[str] = Counter()
        for tok in blk:
            if tok.startswith('[') and tok.endswith(']'):
                opts = tok[1:-1].split('|')
                w = 1 / len(opts)
                scores.update({o: w for o in opts})
            else:
                scores[tok] += 1
        max_v = max(scores.values()) if scores else 0
        winners = sorted(c for c, v in scores.items() if v == max_v)
        out.append(winners[0] if len(winners) == 1 else f"[{'|'.join(winners)}]")
    return ''.join(out)

# -----------------------------------------------------------------------------
# Decode pipeline
# -----------------------------------------------------------------------------

def decode_file(path: str | Path, redundancy: int = CHAR_LEVEL_REDUNDANCY, *, debug: bool = False) -> str:
    rate, wav_data = wav.read(path)
    if rate != SR:
        raise ValueError(f"Expected {SR} Hz wav, got {rate}")
    if wav_data.ndim > 1:
        wav_data = wav_data[:, 0]
    wav_data = wav_data.astype(np.float32)

    spec, freqs = _stft_mag(wav_data, rate)
    tbins = _tone_bins(freqs)

    if debug:
        print("Binarised frames for each STFT column:")
        for c in range(spec.shape[1]):
            print(f"{c}:", ''.join(map(str, _frame_bits(spec, c, tbins))))

    pre_s, pre_e = _find_marker_fwd(spec, tbins, search_from=0)
    post_s, post_e = _find_marker_rev(spec, tbins, search_to=spec.shape[1] - 1)

    if post_s <= pre_e:
        raise RuntimeError("Post‑amble located before pre‑amble – check marker detection")

    data_cols = range(pre_e + 1, post_s)
    sym_cols  = data_cols[::SYMBOL_HOP_FRAMES]

    char_stream = [_symbols_from_bits(_frame_bits(spec, c, tbins)) for c in sym_cols]
    return [_vote(char_stream, redundancy).replace("^", "")]

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser(description="Fuzzy GGWave decoder")
    ap.add_argument("wav", help="input WAV file (48 kHz)")
    ap.add_argument("redundancy", nargs="?", default=4, type=int,
                    help="redundant repeats per char (default 4)")
    ap.add_argument("--debug", action="store_true", help="dump every binarised STFT column")
    args = ap.parse_args()

    msg = decode_file(args.wav, args.redundancy, debug=args.debug)
    print("Decoded:", msg)

if __name__ == "__main__":
    _cli()
