import csv
import math
import wave
from pathlib import Path
from typing import Tuple

import numpy as np


# ============================================================
# Configuration
# ============================================================

FS_OUT = 48000
NOMINAL_BITS = 32
OUTPUT_SECONDS = 2.0

OVERSAMPLINGS = [2, 4, 6, 8, 10]
PAYLOAD_TYPES = ["near_silence", "sine", "noise"]
CONDITIONS = ["clean", "jitter", "bit_slip", "wrong_word_length"]
REPETITIONS = [1, 2, 3, 4, 5]

OUTPUT_DIR = Path("i2s_sonification_dataset")


# ============================================================
# Payload generation
# ============================================================

def generate_payload(payload_type: str, num_words: int, repetition: int, fs: int = FS_OUT) -> np.ndarray:
    """
    Generate synthetic float32 payload samples in [-1, 1].

    payload_type:
        - near_silence: silence but wiht dither, noise floor and quant noise modelled
        - sine: simple sine with frequency varied by repetition
        - noise: white noise with deterministic seed
    """
    t = np.arange(num_words, dtype=np.float64) / fs

    if payload_type == "near_silence":
        # Near-silence model:
        # - very low noise floor
        # - tiny triangular dither
        # - quantization to emulate low-level quantization residue

        rng = np.random.default_rng(seed=2000 + repetition)

        # Low broadband noise floor
        noise_floor = rng.normal(0.0, 3e-5, size=num_words).astype(np.float32)

        # TPDF-style dither approximation: difference of two uniforms
        dither_amp = 1e-5
        dither = (
            rng.uniform(-dither_amp, dither_amp, size=num_words).astype(np.float32)
            + rng.uniform(-dither_amp, dither_amp, size=num_words).astype(np.float32)
        )

        # Combine floor + dither
        low_level = noise_floor + dither

        # Quantize to simulate low-level quantization effects
        # Using 16-bit equivalent quantization residue as a practical model
        q = 1.0 / 32768.0
        data = np.round(low_level / q) * q

        # Keep safely in range
        data = np.clip(data, -1.0, 1.0).astype(np.float32)

    elif payload_type == "sine":
        freqs = [220.0, 440.0, 880.0, 1320.0, 1760.0] #one for each repetition
        freq = freqs[(repetition - 1) % len(freqs)]
        amp = 0.7
        data = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    elif payload_type == "noise":
        rng = np.random.default_rng(seed=1000 + repetition)
        data = rng.normal(0.0, 0.25, size=num_words).astype(np.float32)
        data = np.clip(data, -0.95, 0.95)

    else:
        raise ValueError("Unknown payload type: {}".format(payload_type))

    return data


# ============================================================
# Bit extraction
# ============================================================

def float_to_int32(data: np.ndarray) -> np.ndarray:
    """
    Convert float audio in [-1, 1) to signed 32-bit integer range.
    """
    clipped = np.clip(data, -1.0, 0.99999994)
    return (clipped * 2147483647.0).astype(np.int32)


def extract_serial_bits(samples_int: np.ndarray, tx_bits: int) -> np.ndarray:
    """
    Extract tx_bits from each int32 sample.

    For tx_bits == 32:
        use bit positions 31..0

    For tx_bits == 24:
        use the top 24 bits, i.e. positions 31..8
        This makes the wrong-word-length condition a real framing mismatch:
        24-bit words placed against a 32-bit WS framing monitor.
    """
    if tx_bits == 32:
        bit_positions = np.arange(31, -1, -1, dtype=np.int64)
    elif tx_bits == 24:
        bit_positions = np.arange(31, 7, -1, dtype=np.int64)
    else:
        raise ValueError("Only tx_bits of 24 or 32 are supported in this script.")

    mask = (1 << bit_positions).astype(np.uint32)
    bits = (samples_int[:, None].astype(np.uint32) & mask) > 0
    return bits


# ============================================================
# Core rendering
# ============================================================

def make_nominal_i2s_lines(
    payload: np.ndarray,
    oversample: int,
    condition: str,
    target_num_output_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create WS, SCK, SD line waveforms for a given payload and condition.

    Conditions:
        clean
        jitter
        bit_slip
        wrong_word_length

    wrong_word_length:
        serializes 24-bit words while WS framing remains 32-bit.
    """
    tx_bits = 24 if condition == "wrong_word_length" else 32

    samples_int = float_to_int32(payload)
    sd_bits = extract_serial_bits(samples_int, tx_bits)

    sd_raw = sd_bits.flatten().astype(np.float32) * 2.0 - 1.0
    sd_stretched = np.repeat(sd_raw, oversample)

    delay_bits = 2 if condition == "bit_slip" else 1
    delay_samples = delay_bits * oversample

    sd_line = np.zeros_like(sd_stretched)
    if delay_samples < len(sd_stretched):
        sd_line[delay_samples:] = sd_stretched[:-delay_samples]

    monitor_bits = NOMINAL_BITS

    sck_pattern = np.array([1.0] * oversample + [-1.0] * oversample, dtype=np.float32)
    sck_line = np.tile(sck_pattern, (len(sd_stretched) // len(sck_pattern)) + 1)[: len(sd_stretched)]

    ws_pattern = np.repeat(np.array([1.0, -1.0], dtype=np.float32), monitor_bits * oversample)
    ws_line = np.tile(ws_pattern, (len(sd_stretched) // len(ws_pattern)) + 1)[: len(sd_stretched)]

    if condition == "jitter":
        ws_line, sck_line, sd_line = apply_time_jitter(ws_line, sck_line, sd_line, oversample)

    ws_line = fit_to_length(ws_line, target_num_output_samples)
    sck_line = fit_to_length(sck_line, target_num_output_samples)
    sd_line = fit_to_length(sd_line, target_num_output_samples)

    return ws_line, sck_line, sd_line


def apply_time_jitter(
    ws_line: np.ndarray,
    sck_line: np.ndarray,
    sd_line: np.ndarray,
    oversample: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a small smooth timing warp to all three lines.

    This models timing jitter in the sonified result while preserving
    line-to-line synchrony.
    """
    n = len(sd_line)
    rng = np.random.default_rng(seed=5000 + oversample + n)

    raw = rng.normal(0.0, 0.35 * oversample, size=n).astype(np.float32)

    kernel_len = max(5, 2 * oversample + 1)
    kernel = np.ones(kernel_len, dtype=np.float32) / kernel_len
    smooth = np.convolve(raw, kernel, mode="same")

    t = np.arange(n, dtype=np.float32)
    t_warp = np.clip(t + smooth, 0, n - 1)

    ws_j = np.interp(t, t_warp, ws_line).astype(np.float32)
    sck_j = np.interp(t, t_warp, sck_line).astype(np.float32)
    sd_j = np.interp(t, t_warp, sd_line).astype(np.float32)

    ws_j = np.clip(ws_j, -1.0, 1.0)
    sck_j = np.clip(sck_j, -1.0, 1.0)
    sd_j = np.clip(sd_j, -1.0, 1.0)

    return ws_j, sck_j, sd_j


def fit_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Trim or zero-pad to target length.
    """
    if len(x) == target_len:
        return x.astype(np.float32)
    if len(x) > target_len:
        return x[:target_len].astype(np.float32)

    out = np.zeros(target_len, dtype=np.float32)
    out[: len(x)] = x
    return out


def mix_and_save(
    ws_line: np.ndarray,
    sck_line: np.ndarray,
    sd_line: np.ndarray,
    output_path: Path,
    fs_out: int = FS_OUT,
) -> None:
    """
    Mix to stereo exactly in the spirit of your original code:
      left  = WS + SCK
      right = SD
    """
    left_channel = (0.02 * ws_line) + (0.03 * sck_line)
    right_channel = sd_line * 0.05

    stereo = np.stack((left_channel, right_channel), axis=-1)
    final_pcm = np.clip(stereo, -1.0, 1.0)
    final_pcm = (final_pcm * 32767.0).astype(np.int16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as f_out:
        f_out.setnchannels(2)
        f_out.setsampwidth(2)
        f_out.setframerate(fs_out)
        f_out.writeframes(final_pcm.tobytes())


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(output_dir: Path = OUTPUT_DIR, output_seconds: float = OUTPUT_SECONDS) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_num_output_samples = int(round(FS_OUT * output_seconds))

    manifest_rows = []

    count = 0
    total_expected = len(CONDITIONS) * len(PAYLOAD_TYPES) * len(REPETITIONS) * len(OVERSAMPLINGS)

    for condition in CONDITIONS:
        for payload_type in PAYLOAD_TYPES:
            for repetition in REPETITIONS:
                for oversample in OVERSAMPLINGS:
                    tx_bits = 24 if condition == "wrong_word_length" else 32
                    num_words = int(math.ceil(float(target_num_output_samples) / float(tx_bits * oversample)))

                    payload = generate_payload(payload_type, num_words, repetition, fs=FS_OUT)
                    ws_line, sck_line, sd_line = make_nominal_i2s_lines(
                        payload=payload,
                        oversample=oversample,
                        condition=condition,
                        target_num_output_samples=target_num_output_samples,
                    )

                    filename = (
                        "i2s_{0}__payload-{1}__rep-{2}__os-{3}.wav".format(
                            condition, payload_type, repetition, oversample
                        )
                    )
                    output_path = output_dir / filename

                    mix_and_save(ws_line, sck_line, sd_line, output_path)

                    manifest_rows.append({
                        "filename": filename,
                        "condition": condition,
                        "payload": payload_type,
                        "repetition": repetition,
                        "oversample": oversample,
                        "output_seconds": output_seconds,
                        "fs_out": FS_OUT,
                    })

                    count += 1
                    print("[{0:03d}/{1}] Wrote {2}".format(count, total_expected, output_path))

    manifest_path = output_dir / "manifest.csv"
    with open(str(manifest_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "condition",
                "payload",
                "repetition",
                "oversample",
                "output_seconds",
                "fs_out",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nDone. Generated {0} sonifications in: {1}".format(count, output_dir))
    print("Manifest: {0}".format(manifest_path))


if __name__ == "__main__":
    generate_dataset()
