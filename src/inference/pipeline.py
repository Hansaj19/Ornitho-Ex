"""
Ornitho-Ex | Audio Processing Pipeline
Same preprocessing as bird_sound_preprocessing.ipynb:
  - Load audio → resample to 32 kHz mono
  - Pad / trim to 4-second window (128 000 samples)
  - Log-Mel spectrogram  (128 mel bins × 250 time frames)
  - Programmatic acoustic-concept extraction (6 features)
"""

from __future__ import annotations

import io
import numpy as np
import torch
import librosa
import librosa.feature
from scipy.signal import hilbert


# ── Constants (match species_config.py) ──────────────────────────────────────
SAMPLE_RATE         = 32_000
WINDOW_DURATION_SEC = 4.0
TARGET_SAMPLES      = int(SAMPLE_RATE * WINDOW_DURATION_SEC)   # 128 000
N_MELS              = 128
N_FFT               = 2_048
HOP_LENGTH          = 512
TARGET_FRAMES       = 250   # ceil(128 000 / 512)


# ── Audio Loading ─────────────────────────────────────────────────────────────

def load_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """Load raw bytes (wav/mp3/ogg/flac) → mono float32 waveform at 32 kHz."""
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


def load_audio_path(path: str) -> np.ndarray:
    """Load an audio file from disk → mono float32 waveform at 32 kHz."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


# ── Segmentation ──────────────────────────────────────────────────────────────

def extract_best_segment(y: np.ndarray) -> np.ndarray:
    """
    Pick the 4-second window with the highest RMS energy (most bird activity).
    If the clip is shorter than 4 s, zero-pad on the right.
    """
    if len(y) <= TARGET_SAMPLES:
        return np.pad(y, (0, TARGET_SAMPLES - len(y)))

    # Slide over all possible 4-s windows and pick the loudest
    rms_vals = []
    for start in range(0, len(y) - TARGET_SAMPLES + 1, HOP_LENGTH):
        rms_vals.append(np.sqrt(np.mean(y[start: start + TARGET_SAMPLES] ** 2)))

    best_start = np.argmax(rms_vals) * HOP_LENGTH
    return y[best_start: best_start + TARGET_SAMPLES]


# ── Log-Mel Spectrogram ───────────────────────────────────────────────────────

def compute_log_mel(y: np.ndarray) -> np.ndarray:
    """
    Return a (128, 250) log-mel spectrogram (float32) from a 4-s waveform.
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # Ensure exactly TARGET_FRAMES columns
    if log_mel.shape[1] < TARGET_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, TARGET_FRAMES - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :TARGET_FRAMES]

    return log_mel   # (128, 250)


def spectrogram_to_tensor(log_mel: np.ndarray) -> torch.Tensor:
    """
    Format the float32 log-mel spectrogram for model inference.
    Returns (1, 1, 128, 250) float32 tensor matching training format.
    """
    return torch.from_numpy(log_mel.astype(np.float32)).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)


# ── Acoustic Concept Extraction (Programmatic, zero human annotation) ─────────

def extract_acoustic_concepts(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Deterministically compute the 6 physical acoustic features used as
    CBM ground-truth concepts during training.

    Returns
    -------
    dict with keys:
        peak_frequency      Hz   — dominant pitch
        trill_rate          Hz   — note-repetition rate
        call_duration       0–1  — fraction of active vocalisation
        fm_rate             Hz/frame — frequency modulation depth
        spectral_centroid   Hz   — spectral centre of mass
        inter_call_silence  0–1  — silent-fraction (gaps between calls)
    """
    # ── peak frequency via FFT ────────────────────────────────────────────────
    fft_mag  = np.abs(np.fft.rfft(y, n=N_FFT))
    freqs    = np.fft.rfftfreq(N_FFT, d=1.0 / sr)
    peak_frequency = float(freqs[np.argmax(fft_mag)])

    # ── spectral centroid (librosa) ───────────────────────────────────────────
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectral_centroid = float(cent.mean())

    # ── call duration via amplitude envelope ──────────────────────────────────
    envelope  = np.abs(hilbert(y))
    threshold = envelope.mean() + 0.5 * envelope.std()
    active    = (envelope > threshold).astype(float)
    call_duration = float(active.mean())

    # ── inter-call silence ────────────────────────────────────────────────────
    inter_call_silence = 1.0 - call_duration

    # ── trill rate: zero-crossing bursts in a 50-ms window ───────────────────
    win_samples  = int(0.05 * sr)
    zcr          = librosa.feature.zero_crossing_rate(y, frame_length=win_samples, hop_length=win_samples // 2)[0]
    trill_rate   = float(zcr.mean() * sr / win_samples)

    # ── FM rate: mean absolute pitch derivative from pyin ─────────────────────
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=200, fmax=12_000, sr=sr, hop_length=HOP_LENGTH
        )
        if f0 is not None and voiced_flag is not None:
            voiced_f0 = f0[voiced_flag]
            fm_rate   = float(np.mean(np.abs(np.diff(voiced_f0)))) if len(voiced_f0) > 1 else 0.0
        else:
            fm_rate = 0.0
    except Exception:
        fm_rate = 0.0

    return {
        "peak_frequency":     peak_frequency,
        "trill_rate":         trill_rate,
        "call_duration":      call_duration,
        "fm_rate":            fm_rate,
        "spectral_centroid":  spectral_centroid,
        "inter_call_silence": inter_call_silence,
    }


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def process_audio(audio_input) -> dict:
    """
    Run the full preprocessing pipeline on an audio input.

    Parameters
    ----------
    audio_input : bytes or str
        Raw audio bytes (from file uploader / microphone) or a file path.

    Returns
    -------
    dict with keys:
        waveform    np.ndarray (N,)        raw waveform at 32 kHz
        segment     np.ndarray (128 000,)  best 4-s segment
        log_mel     np.ndarray (128, 250)  log-mel spectrogram
        tensor      torch.Tensor (1,1,128,250)
        concepts    dict                   6 acoustic features
    """
    if isinstance(audio_input, bytes):
        y = load_audio_bytes(audio_input)
    else:
        y = load_audio_path(str(audio_input))

    segment  = extract_best_segment(y)
    log_mel  = compute_log_mel(segment)
    tensor   = spectrogram_to_tensor(log_mel)
    concepts = extract_acoustic_concepts(segment)

    return {
        "waveform": y,
        "segment":  segment,
        "log_mel":  log_mel,
        "tensor":   tensor,
        "concepts": concepts,
    }
