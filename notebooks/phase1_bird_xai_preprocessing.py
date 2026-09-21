# =============================================================================
# ORNITHO-EX | Notebook 1: bird-xai-preprocessing
# Phase 1 — Dataset Acquisition (BirdCLEF+ 2026 + NIPS4Bplus)
# =============================================================================
# IMPORTANT: Before running this notebook:
#   1. Internet: NOT required (audio reads from /kaggle/input/ — no downloads)
#   2. GPU accelerator: OFF (CPU-only notebook — saves GPU quota)
#   3. Attach the following datasets in the Kaggle sidebar:
#      a. BirdCLEF+ 2026 (competition)  →  mounts at /kaggle/input/birdclef-plus-2026/
#      b. Multi-label Bird Species Classification (NIPS4Bplus)
#              →  mounts at /kaggle/input/competitions/multilabel-bird-species-classification-nips2013/
#   4. Runtime: Run all cells top-to-bottom
# =============================================================================


# ─────────────────────────────────────────────
# CELL 1 — Install / upgrade required packages
# ─────────────────────────────────────────────
# %%
import subprocess, sys

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

pip_install(
    "librosa==0.10.2",
    "soundfile",
    "scipy",
    "tqdm",
    "pandas",
    "numpy",
)

print("✅ Dependencies installed.")


# ─────────────────────────────────────────────
# CELL 2 — Imports & logging
# ─────────────────────────────────────────────
# %%
import os
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import hilbert
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ornitho-ex")

print("✅ Imports done.")


# ─────────────────────────────────────────────
# CELL 3 — Paths & configuration
# ─────────────────────────────────────────────
# %%

# ── Input: BirdCLEF+ 2026 (read-only) ────────────────────────────
# Kaggle competition slug: birdclef-2026
# Actual mount path confirmed: /kaggle/input/competitions/birdclef-2026/
# Folder structure:
#   train_audio/<primary_label>/<filename>.ogg  ← focal recordings
#     NOTE: filename in train.csv already contains the sub-path,
#           e.g.  filename = "houspa/XC977863.ogg"
#           so audio_path = train_audio / filename  (no extra primary_label prefix)
#   train.csv                         ← metadata
#   taxonomy.csv
BIRDCLEF_DIR       = Path("/kaggle/input/competitions/birdclef-2026")
BIRDCLEF_METADATA  = BIRDCLEF_DIR / "train.csv"
BIRDCLEF_AUDIO_DIR = BIRDCLEF_DIR / "train_audio"

# ── Input: NIPS4Bplus (read-only) ───────────────────────────────
# Kaggle slug: competitions/multilabel-bird-species-classification-nips2013
# Files visible:
#   NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS*.csv  ← per-recording annotation CSVs
#   NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV*    ← WAV audio files
#   NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_M*      ← metadata
NIPS4B_ROOT    = Path("/kaggle/input/competitions/multilabel-bird-species-classification-nips2013")
NIPS4B_WAV_DIR = NIPS4B_ROOT   # WAV files live at root level
NIPS4B_CSV_DIR = NIPS4B_ROOT   # annotation CSVs also at root level

# ── Output: processed features (writable, 20 GB limit) ──────
BASE_OUT           = Path("/kaggle/working")
PROC_DIR           = BASE_OUT / "processed_features"
BIRDCLEF_PROC_DIR  = PROC_DIR / "birdclef"     # train/val/test NPZ files
NIPS4B_PROC_DIR    = PROC_DIR / "nips4bplus" / "validation"  # validation NPZ files
NIPS4B_CUT_DIR     = BASE_OUT / "nips4bplus_cut_wavs"  # temporary cut WAVs

for d in [
    BIRDCLEF_PROC_DIR / "train",
    BIRDCLEF_PROC_DIR / "val",
    BIRDCLEF_PROC_DIR / "test",
    NIPS4B_PROC_DIR,
    NIPS4B_CUT_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ── Feature extraction settings ─────────────────────────────
SAMPLE_RATE         = 32000    # Hz — native rate of BirdCLEF 2021 .ogg files
WINDOW_SEC          = 4.0      # seconds per segment
N_MELS              = 128
N_FFT               = 2048
HOP_LENGTH          = 512
# Expected spectrogram width at 32 kHz: ceil(32000 * 4 / 512) = 250 frames
SPEC_FRAMES         = int(np.ceil(SAMPLE_RATE * WINDOW_SEC / HOP_LENGTH))

# ── Dataset balancing ────────────────────────────────────────
MIN_RATING          = 3.5      # BirdCLEF quality star filter (0–5)
MAX_CLIPS_PER_SP    = 50       # cap per species for class balance
TRAIN_RATIO         = 0.80
VAL_RATIO           = 0.10
TEST_RATIO          = 0.10     # must sum to 1.0

# ── Disk safety ──────────────────────────────────────────────
MAX_DISK_GB         = 2.0      # assert total /kaggle/working usage < this

print("✅ Configuration set.")
print(f"   Dataset:  BirdCLEF+ 2026  |  mount: /kaggle/input/competitions/birdclef-2026")
print(f"   Metadata: {BIRDCLEF_METADATA}")
print(f"   Audio:    {BIRDCLEF_AUDIO_DIR}")
print(f"   NIPS4B:   {NIPS4B_ROOT}")
print(f"   SAMPLE_RATE={SAMPLE_RATE} Hz  |  WINDOW={WINDOW_SEC}s  |  FRAMES≈{SPEC_FRAMES}")
print(f"   N_MELS={N_MELS}  |  N_FFT={N_FFT}  |  HOP={HOP_LENGTH}")
print(f"   MIN_RATING≥{MIN_RATING}  |  MAX_CLIPS/species={MAX_CLIPS_PER_SP}")
print(f"   Split: {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / {TEST_RATIO:.0%} test")

# ── Quick path sanity check ────────────────────────────────────
print("\n── Path sanity check ──")
print(f"  BirdCLEF+ 2026 root:   {'EXISTS' if BIRDCLEF_DIR.exists() else 'NOT FOUND ⚠️'}")
print(f"  train.csv:             {'EXISTS' if BIRDCLEF_METADATA.exists() else 'NOT FOUND ⚠️'}")
print(f"  train_audio/:          {'EXISTS' if BIRDCLEF_AUDIO_DIR.exists() else 'NOT FOUND ⚠️'}")
print(f"  NIPS4B root:           {'EXISTS' if NIPS4B_ROOT.exists() else 'NOT FOUND ⚠️'}")
if BIRDCLEF_DIR.exists():
    top_level = [p.name for p in BIRDCLEF_DIR.iterdir()]
    print(f"  BirdCLEF+ contents:    {top_level}")
if NIPS4B_ROOT.exists():
    nips_top = [p.name for p in NIPS4B_ROOT.iterdir()][:10]
    print(f"  NIPS4B contents:       {nips_top}")


# ─────────────────────────────────────────────
# CELL 4 — Species lists & taxonomy mapping
# ─────────────────────────────────────────────
# %%

# Core: NIPS4Bplus 51 European species (scientific names, 2013 taxonomy)
# These are the species for which faithfulness evaluation is possible.
NIPS4BPLUS_SPECIES = [
    "Acrocephalus arundinaceus",
    "Acrocephalus scirpaceus",
    "Alauda arvensis",
    "Anthus pratensis",
    "Anthus trivialis",
    "Buteo buteo",
    "Carduelis carduelis",
    "Carduelis chloris",
    "Carduelis spinus",
    "Certhia brachydactyla",
    "Certhia familiaris",
    "Coccothraustes coccothraustes",
    "Columba palumbus",
    "Corvus corone",
    "Cuculus canorus",
    "Emberiza citrinella",
    "Erithacus rubecula",
    "Ficedula hypoleuca",
    "Fringilla coelebs",
    "Garrulus glandarius",
    "Hippolais icterina",
    "Jynx torquilla",
    "Lanius collurio",
    "Luscinia megarhynchos",
    "Luscinia svecica",
    "Motacilla alba",
    "Motacilla flava",
    "Muscicapa striata",
    "Oriolus oriolus",
    "Parus ater",
    "Parus caeruleus",
    "Parus major",
    "Passer domesticus",
    "Passer montanus",
    "Phoenicurus ochruros",
    "Phoenicurus phoenicurus",
    "Phylloscopus collybita",
    "Phylloscopus trochilus",
    "Picus viridis",
    "Regulus ignicapilla",
    "Regulus regulus",
    "Saxicola rubetra",
    "Saxicola torquata",
    "Sitta europaea",
    "Streptopelia decaocto",
    "Sturnus vulgaris",
    "Sylvia atricapilla",
    "Sylvia borin",
    "Sylvia communis",
    "Troglodytes troglodytes",
    "Turdus merula",
]

# Extended European species — additional training coverage in BirdCLEF 2021
EXTENDED_SPECIES = [
    "Turdus philomelos",
    "Turdus viscivorus",
    "Sylvia curruca",
    "Locustella naevia",
    "Acrocephalus palustris",
    "Emberiza schoeniclus",
    "Parus palustris",
    "Dendrocopos major",
    "Dryocopus martius",
    "Anthus spinoletta",
    "Oenanthe oenanthe",
    "Hirundo rustica",
    "Delichon urbicum",
    "Apus apus",
    "Upupa epops",
    "Alcedo atthis",
    "Merops apiaster",
    "Caprimulgus europaeus",
    "Sylvia nisoria",
    "Lullula arborea",
]

NIPS4BPLUS_TARGET_SPECIES = NIPS4BPLUS_SPECIES + EXTENDED_SPECIES  # kept for overlap check

# ── Taxonomy mapping: NIPS4Bplus 2013 names → modern scientific names ──────
# Used ONLY for the NIPS4Bplus overlap check at the end of the notebook.
# BirdCLEF+ 2026 may not contain these European species at all;
# we train on ALL available BirdCLEF 2026 species (see Cell 5).
TAXONOMY_MAP = {
    "Carduelis chloris" : "Chloris chloris",
    "Carduelis spinus"  : "Spinus spinus",
    "Parus ater"        : "Periparus ater",
    "Parus caeruleus"   : "Cyanistes caeruleus",
    "Parus palustris"   : "Poecile palustris",
    "Saxicola torquata" : "Saxicola rubicola",
    "Sylvia communis"   : "Curruca communis",
    "Sylvia curruca"    : "Curruca curruca",
    "Sylvia nisoria"    : "Curruca nisoria",
    "Luscinia svecica"  : "Cyanecula svecica",
}

# Build reverse lookup: modern NIPS4Bplus name → canonical 2013 name
# (used to check if a BirdCLEF species appears in NIPS4Bplus validation set)
MODERN_TO_NIPS4B_CANONICAL = {}
for sp in NIPS4BPLUS_TARGET_SPECIES:
    modern = TAXONOMY_MAP.get(sp, sp)
    MODERN_TO_NIPS4B_CANONICAL[modern] = sp

NIPS4B_MODERN_NAMES = set(MODERN_TO_NIPS4B_CANONICAL.keys())

print(f"✅ Species lists ready.")
print(f"   NIPS4Bplus target (validation overlap check): {len(NIPS4BPLUS_SPECIES)}")
print(f"   Extended European:                            {len(EXTENDED_SPECIES)}")
print(f"   Total NIPS4Bplus target (incl. extended):     {len(NIPS4BPLUS_TARGET_SPECIES)}")
print(f"   Taxonomy remaps:                              {len(TAXONOMY_MAP)}")
print(f"   NOTE: Cell 5 will train on ALL BirdCLEF 2026 species (not just these).")


# ─────────────────────────────────────────────
# CELL 5 — Load BirdCLEF+ 2026 metadata & build manifest
#
# Strategy: use ALL species in BirdCLEF 2026 (with quality filter).
# Do NOT pre-filter by European target species — BirdCLEF 2026 covers
# global species and most European NIPS4Bplus species are NOT present.
# NIPS4Bplus overlap is computed afterwards (Cell 10) as the validation
# subset — this is the correct two-list strategy.
# ─────────────────────────────────────────────
# %%
import random
random.seed(42)
np.random.seed(42)

print(f"Loading BirdCLEF+ 2026 metadata from: {BIRDCLEF_METADATA}")
assert BIRDCLEF_METADATA.exists(), (
    f"❌ Metadata file not found: {BIRDCLEF_METADATA}\n"
    "   → In Kaggle, click '+ Add Data' → Competitions → 'BirdCLEF 2026'.\n"
    "   → Confirmed mount path: /kaggle/input/competitions/birdclef-2026/"
)

bc_meta = pd.read_csv(BIRDCLEF_METADATA)
print(f"   Loaded {len(bc_meta):,} rows, {bc_meta['primary_label'].nunique()} unique species.")
print(f"   Columns: {list(bc_meta.columns)}")

# ── Detect column names dynamically (robust to minor CSV schema changes) ──
SCI_NAME_COL = next(
    (c for c in bc_meta.columns if c.lower() in ("scientific_name", "sci_name", "species")),
    None
)
LABEL_COL = next(
    (c for c in bc_meta.columns if c.lower() in ("primary_label", "label", "ebird_code")),
    None
)
FILENAME_COL = next(
    (c for c in bc_meta.columns if c.lower() in ("filename", "file_name", "filepath")),
    None
)
RATING_COL = next(
    (c for c in bc_meta.columns if c.lower() in ("rating", "quality", "quality_rating")),
    None
)

for name, val in [("scientific_name", SCI_NAME_COL), ("primary_label", LABEL_COL),
                  ("filename", FILENAME_COL), ("rating", RATING_COL)]:
    status = f"→ '{val}'" if val else "NOT FOUND ⚠️"
    print(f"   {name:20s}: {status}")

assert SCI_NAME_COL  is not None, f"❌ No scientific_name col. Cols: {list(bc_meta.columns)}"
assert LABEL_COL     is not None, f"❌ No primary_label col.  Cols: {list(bc_meta.columns)}"
assert FILENAME_COL  is not None, f"❌ No filename col.       Cols: {list(bc_meta.columns)}"

# ── Step 1: Quality filter ────────────────────────────────────────
# Keep all species; just filter by rating to prefer cleaner recordings.
bc_filtered = bc_meta.copy()
if RATING_COL is not None:
    n_before = len(bc_filtered)
    bc_filtered = bc_filtered.dropna(subset=[RATING_COL])
    bc_filtered = bc_filtered[bc_filtered[RATING_COL] >= MIN_RATING].copy()
    print(f"\n   After rating≥{MIN_RATING} filter: {len(bc_filtered):,} rows "
          f"({n_before - len(bc_filtered):,} removed), "
          f"{bc_filtered[SCI_NAME_COL].nunique()} species.")
else:
    print(f"   ⚠️  No rating column found — using all {len(bc_filtered):,} rows.")

# ── Step 2: Build audio path ──────────────────────────────────────
# IMPORTANT: In BirdCLEF+ 2026, the 'filename' column ALREADY contains
# the subdirectory, e.g. filename = "houspa/XC977863.ogg".
# Correct path: AUDIO_DIR / filename   (NOT AUDIO_DIR / primary_label / filename)
bc_filtered = bc_filtered.copy()
bc_filtered["audio_path"] = bc_filtered[FILENAME_COL].apply(
    lambda fn: str(BIRDCLEF_AUDIO_DIR / fn)
)
# Use scientific_name as the canonical training label
bc_filtered["canonical_name"] = bc_filtered[SCI_NAME_COL]

# ── Step 3: Cap at MAX_CLIPS_PER_SP per species ───────────────────
balanced_rows = []
for sp, grp in bc_filtered.groupby(SCI_NAME_COL):
    if RATING_COL and len(grp) > MAX_CLIPS_PER_SP:
        grp = grp.sort_values(RATING_COL, ascending=False).head(MAX_CLIPS_PER_SP)
    balanced_rows.append(grp)

manifest_df = pd.concat(balanced_rows, ignore_index=True)

# ── Step 4: Stratified 80/10/10 train/val/test split ─────────────
manifest_df["split"] = "train"

for sp, grp in manifest_df.groupby("canonical_name"):
    idx = grp.index.tolist()
    if len(idx) < 3:
        continue  # too few — keep all in train
    n_val  = max(1, int(len(idx) * VAL_RATIO))
    n_test = max(1, int(len(idx) * TEST_RATIO))
    random.shuffle(idx)
    manifest_df.loc[idx[:n_val],          "split"] = "val"
    manifest_df.loc[idx[n_val:n_val+n_test], "split"] = "test"

# ── Step 5: Summary ───────────────────────────────────────────────
accepted_species = sorted(manifest_df["canonical_name"].unique().tolist())

print(f"\n{'='*60}")
print(f"BIRDCLEF+ 2026 MANIFEST SUMMARY")
print(f"{'='*60}")
print(f"  Species in dataset:  {len(accepted_species)}")
print(f"  Total clips:         {len(manifest_df)}")
print(f"  Train:               {(manifest_df['split']=='train').sum()}")
print(f"  Val:                 {(manifest_df['split']=='val').sum()}")
print(f"  Test:                {(manifest_df['split']=='test').sum()}")

# Quick check: how many NIPS4Bplus target species appear?
nips_overlap_in_bc = [s for s in NIPS4B_MODERN_NAMES if s in set(accepted_species)]
print(f"\n  NIPS4Bplus overlap in BirdCLEF 2026: {len(nips_overlap_in_bc)} species")
if nips_overlap_in_bc:
    for s in sorted(nips_overlap_in_bc):
        print(f"    ✓ {s}")
else:
    print(f"  ℹ️  No direct NIPS4Bplus species overlap — XAI faithfulness evaluation")
    print(f"     will use NIPS4Bplus recordings independently (not class-matched).")

# ── Step 6: Verify a sample audio path exists ─────────────────────
sample_row = manifest_df.iloc[0]
if Path(sample_row["audio_path"]).exists():
    print(f"\n  ✅ Audio path check OK: {sample_row['audio_path']}")
else:
    print(f"\n  ⚠️  Sample audio path NOT found: {sample_row['audio_path']}")
    print(f"     Filename from CSV: {sample_row[FILENAME_COL]}")
    print(f"     Audio dir:        {BIRDCLEF_AUDIO_DIR}")

# Save manifest for audit
manifest_path = PROC_DIR / "birdclef_manifest.csv"
manifest_df.to_csv(manifest_path, index=False)
print(f"\n✅ Manifest saved → {manifest_path}")


# ─────────────────────────────────────────────
# CELL 6 — Acoustic concept extraction helpers
# ─────────────────────────────────────────────
# %%

def compute_log_mel(
    audio: np.ndarray,
    sr: int,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    target_frames: int = SPEC_FRAMES,
) -> np.ndarray:
    """
    Compute a log-mel spectrogram from a mono float32 audio array.

    Pads or truncates the spectrogram to `target_frames` columns.
    Returns: float16 array of shape (n_mels, target_frames).
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # Pad or truncate to fixed width
    if log_mel.shape[1] < target_frames:
        pad_width = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant",
                         constant_values=log_mel.min())
    else:
        log_mel = log_mel[:, :target_frames]

    return log_mel.astype(np.float16)


def compute_acoustic_concepts(
    audio: np.ndarray,
    sr: int,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> dict:
    """
    Extract 6 programmatic acoustic concept targets from a mono audio segment.

    Concepts:
      1. peak_frequency      — frequency bin with maximum energy (Hz)
      2. trill_rate          — estimated AM modulation rate (Hz)
      3. call_duration       — fraction of frames exceeding energy threshold
      4. fm_rate             — mean absolute rate of change of spectral centroid
      5. spectral_centroid   — mean spectral centroid across frames (Hz)
      6. inter_call_silence  — fraction of frames below energy threshold (complementary to call_duration)

    Returns: dict of float32 values (one per concept).
    All outputs are verified to be finite; NaN/inf replaced with 0.0.
    """
    if len(audio) == 0:
        return {
            "peak_frequency": 0.0, "trill_rate": 0.0, "call_duration": 0.0,
            "fm_rate": 0.0, "spectral_centroid": 0.0, "inter_call_silence": 0.0,
        }

    # ── 1. Peak frequency ────────────────────────────────────
    fft_mag = np.abs(np.fft.rfft(audio, n=n_fft))
    freqs   = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    peak_frequency = float(freqs[np.argmax(fft_mag)])

    # ── 2. Trill rate (AM modulation rate) ──────────────────
    # Detect amplitude envelope via analytic signal, then estimate
    # dominant oscillation in envelope via autocorrelation.
    envelope = np.abs(hilbert(audio))
    # Downsample envelope to reduce autocorr cost
    env_ds   = envelope[::hop_length]
    if len(env_ds) > 2:
        autocorr = np.correlate(env_ds - env_ds.mean(), env_ds - env_ds.mean(), mode="full")
        autocorr = autocorr[len(autocorr) // 2:]   # keep non-negative lags
        autocorr = autocorr / (autocorr[0] + 1e-9)  # normalize
        # Find first peak after lag=0 (ignore lag=0 itself)
        min_lag = max(1, int(0.02 * sr / hop_length))  # > 20 ms
        max_lag = max(min_lag + 1, int(0.5  * sr / hop_length))  # < 500 ms
        search  = autocorr[min_lag:max_lag]
        if len(search) > 0:
            best_lag = int(np.argmax(search)) + min_lag
            trill_rate = float((sr / hop_length) / best_lag) if best_lag > 0 else 0.0
        else:
            trill_rate = 0.0
    else:
        trill_rate = 0.0

    # ── 3. Call duration (active frame fraction) ─────────────
    rms_frames = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    if rms_frames.max() > 0:
        threshold   = 0.1 * rms_frames.max()   # 10% of peak RMS
        active_mask = rms_frames > threshold
        call_duration     = float(active_mask.mean())
        inter_call_silence = float((~active_mask).mean())
    else:
        call_duration      = 0.0
        inter_call_silence = 1.0

    # ── 4 & 5. Spectral centroid & FM rate ──────────────────
    sc_frames = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    spectral_centroid = float(np.mean(sc_frames))
    # FM rate = mean absolute frame-to-frame change in centroid (Hz/frame)
    fm_rate = float(np.mean(np.abs(np.diff(sc_frames)))) if len(sc_frames) > 1 else 0.0

    concepts = {
        "peak_frequency"    : peak_frequency,
        "trill_rate"        : trill_rate,
        "call_duration"     : call_duration,
        "fm_rate"           : fm_rate,
        "spectral_centroid" : spectral_centroid,
        "inter_call_silence": inter_call_silence,
    }

    # Guard: replace any non-finite value with 0.0
    for k, v in concepts.items():
        if not np.isfinite(v):
            log.warning(f"Non-finite concept {k}={v} — replacing with 0.0")
            concepts[k] = 0.0

    return {k: np.float32(v) for k, v in concepts.items()}


def load_and_segment_audio(
    path: str,
    sr_target: int = SAMPLE_RATE,
    window_sec: float = WINDOW_SEC,
) -> List[np.ndarray]:
    """
    Load a .ogg (or any soundfile-readable) audio file at `sr_target`.
    Converts to mono. Slices into non-overlapping windows of `window_sec`.
    Returns a list of float32 mono audio arrays, each of length sr_target * window_sec.
    Discards the trailing partial window if shorter than 1 second.
    """
    try:
        audio, sr = librosa.load(path, sr=sr_target, mono=True, dtype=np.float32)
    except Exception as e:
        log.warning(f"  Could not load {path}: {e}")
        return []

    window_samples = int(sr_target * window_sec)
    if len(audio) < sr_target:  # shorter than 1 second — skip entirely
        return []

    segments = []
    for start in range(0, len(audio) - window_samples + 1, window_samples):
        seg = audio[start : start + window_samples]
        segments.append(seg)

    return segments


print("✅ Acoustic concept extraction helpers defined.")
print(f"   Spectrogram output shape: ({N_MELS}, {SPEC_FRAMES}) as float16")
print("   Concepts: peak_frequency, trill_rate, call_duration,")
print("             fm_rate, spectral_centroid, inter_call_silence")


# ─────────────────────────────────────────────
# CELL 7 — BirdCLEF feature extraction (local .ogg → .npz)
#
# Zero raw audio written to disk.
# All audio is streamed from read-only /kaggle/input/birdclef-2021/.
# Output: compact .npz files in /kaggle/working/processed_features/birdclef/
# ─────────────────────────────────────────────
# %%

print("Starting BirdCLEF 2021 local feature extraction...")
print(f"  Reading audio from: {BIRDCLEF_AUDIO_DIR}")
print(f"  Writing NPZ to:     {BIRDCLEF_PROC_DIR}")
print()

# Build per-split output dirs and assign numeric class labels
label2idx = {sp: i for i, sp in enumerate(sorted(accepted_species))}
idx2label  = {str(i): sp for sp, i in label2idx.items()}

bc_extraction_log = []  # list of dicts for audit

for split in ["train", "val", "test"]:
    split_df  = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
    split_dir = BIRDCLEF_PROC_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{split.upper()}] {len(split_df)} clips → {split_dir}")

    for _, row in tqdm(split_df.iterrows(), total=len(split_df),
                       desc=f"BirdCLEF {split}", leave=True):
        audio_path  = row["audio_path"]
        canonical   = row["canonical_name"]
        label_idx   = label2idx[canonical]

        if not Path(audio_path).exists():
            log.warning(f"  Audio file missing: {audio_path}")
            bc_extraction_log.append({
                "audio_path": audio_path, "canonical_name": canonical,
                "split": split, "segments_saved": 0, "status": "missing",
            })
            continue

        segments = load_and_segment_audio(audio_path, sr_target=SAMPLE_RATE,
                                          window_sec=WINDOW_SEC)

        if not segments:
            bc_extraction_log.append({
                "audio_path": audio_path, "canonical_name": canonical,
                "split": split, "segments_saved": 0, "status": "no_segments",
            })
            continue

        # One NPZ per source audio file (may contain multiple segments)
        # Shape of log_mels: (n_segs, N_MELS, SPEC_FRAMES) as float16
        # Shape of concepts: (n_segs, 6) as float32
        log_mels = []
        concepts_list = []

        for seg in segments:
            lm = compute_log_mel(seg, sr=SAMPLE_RATE)
            cv = compute_acoustic_concepts(seg, sr=SAMPLE_RATE)
            log_mels.append(lm)
            concepts_list.append(list(cv.values()))

        log_mels_arr   = np.stack(log_mels, axis=0)          # (n_segs, 128, T)
        concepts_arr   = np.array(concepts_list, dtype=np.float32)  # (n_segs, 6)

        # NPZ filename: canonical_name + md5-based stem for uniqueness
        import hashlib
        stem    = hashlib.md5(audio_path.encode()).hexdigest()[:8]
        npz_name = f"{canonical.replace(' ', '_')}_{stem}.npz"
        npz_path = split_dir / npz_name

        np.savez_compressed(
            str(npz_path),
            log_mel=log_mels_arr,                # float16, shape (n_segs, 128, T)
            concepts=concepts_arr,               # float32, shape (n_segs, 6)
            label=np.int32(label_idx),           # scalar
            canonical_name=canonical,            # str (stored as 0-d array)
            source_path=audio_path,              # str
            concept_names=np.array([             # concept key order
                "peak_frequency", "trill_rate", "call_duration",
                "fm_rate", "spectral_centroid", "inter_call_silence"
            ]),
        )

        bc_extraction_log.append({
            "audio_path":    audio_path,
            "canonical_name": canonical,
            "split":          split,
            "segments_saved": len(segments),
            "npz_path":       str(npz_path),
            "status":         "ok",
        })

bc_log_df = pd.DataFrame(bc_extraction_log)
bc_log_df.to_csv(PROC_DIR / "birdclef_extraction_log.csv", index=False)

ok    = (bc_log_df["status"] == "ok").sum()
miss  = (bc_log_df["status"] == "missing").sum()
noseg = (bc_log_df["status"] == "no_segments").sum()

print(f"\n{'='*60}")
print(f"BIRDCLEF FEATURE EXTRACTION COMPLETE")
print(f"{'='*60}")
print(f"  Files processed (OK):       {ok}")
print(f"  Files missing:              {miss}")
print(f"  Files with no segments:     {noseg}")
total_segs = bc_log_df[bc_log_df["status"] == "ok"]["segments_saved"].sum()
print(f"  Total 4-second segments:    {int(total_segs)}")
print(f"  NPZ output dir:             {BIRDCLEF_PROC_DIR}")


# ─────────────────────────────────────────────
# CELL 8 — NIPS4Bplus: extract archives & cut WAVs from annotations
#           (Inline port of cut_nips4bplus_files.py)
#           CRITICAL: timestamp metadata MUST be preserved.
# ─────────────────────────────────────────────
# %%
import tarfile

# ── NIPS4Bplus archive locations (read-only input) ────────────────────────────
# The Kaggle dataset delivers three compressed archives — not pre-extracted.
#   NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV.tar.gz   → WAV audio files
#   NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS.tar          → annotation CSV files (1 per recording)
#   NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_MFCC.tar.gz   → MFCC features (not needed)
NIPS4B_WAV_ARCHIVE    = NIPS4B_ROOT / "NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV.tar.gz"
NIPS4B_LABELS_ARCHIVE = NIPS4B_ROOT / "NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS.tar"

# ── Extraction targets (writable working dir) ─────────────────────────────────
NIPS4B_EXTRACT_DIR = BASE_OUT / "nips4b_extracted"
NIPS4B_WAV_EXTRACT = NIPS4B_EXTRACT_DIR / "wavs"
NIPS4B_CSV_EXTRACT = NIPS4B_EXTRACT_DIR / "labels"
NIPS4B_WAV_EXTRACT.mkdir(parents=True, exist_ok=True)
NIPS4B_CSV_EXTRACT.mkdir(parents=True, exist_ok=True)



def _dir_nonempty(d: Path) -> bool:
    try:
        return any(d.iterdir())
    except StopIteration:
        return False
    except Exception:
        return False

print("=" * 60)
print("NIPS4Bplus — Archive Extraction")
print("=" * 60)
print(f"  Source archives: {NIPS4B_ROOT}")
print(f"  Extract target:  {NIPS4B_EXTRACT_DIR}")
print()

# Check archives exist
for arch in [NIPS4B_WAV_ARCHIVE, NIPS4B_LABELS_ARCHIVE]:
    status = "✅ FOUND" if arch.exists() else "❌ NOT FOUND"
    size   = f"({arch.stat().st_size / 1e6:.0f} MB)" if arch.exists() else ""
    print(f"  {status}  {arch.name}  {size}")
print()

# Extract WAV archive
wav_ok = False
if _dir_nonempty(NIPS4B_WAV_EXTRACT):
    print(f"  ⏭️  WAV archive: already extracted → {NIPS4B_WAV_EXTRACT}")
    wav_ok = True
elif NIPS4B_WAV_ARCHIVE.exists():
    print(f"  📦 Extracting WAV archive "
          f"({NIPS4B_WAV_ARCHIVE.stat().st_size / 1e6:.0f} MB) — may take 1–3 min...")
    try:
        with tarfile.open(str(NIPS4B_WAV_ARCHIVE), "r:gz") as tar:
            tar.extractall(path=str(NIPS4B_WAV_EXTRACT))
        n_wav = sum(1 for _ in NIPS4B_WAV_EXTRACT.rglob("*.wav"))
        print(f"     ✅ Done — {n_wav} WAV files extracted")
        wav_ok = True
    except Exception as e:
        print(f"     ❌ WAV extraction failed: {e}")
else:
    print(f"  ❌ WAV archive not found: {NIPS4B_WAV_ARCHIVE}")

# Extract labels (annotation CSV) archive
csv_ok = False
if _dir_nonempty(NIPS4B_CSV_EXTRACT):
    print(f"  ⏭️  Labels archive: already extracted → {NIPS4B_CSV_EXTRACT}")
    csv_ok = True
elif NIPS4B_LABELS_ARCHIVE.exists():
    print(f"  📦 Extracting labels archive "
          f"({NIPS4B_LABELS_ARCHIVE.stat().st_size / 1e6:.0f} MB)...")
    try:
        with tarfile.open(str(NIPS4B_LABELS_ARCHIVE), "r:") as tar:
            tar.extractall(path=str(NIPS4B_CSV_EXTRACT))
        n_csv = sum(1 for _ in NIPS4B_CSV_EXTRACT.rglob("*.csv"))
        print(f"     ✅ Done — {n_csv} CSV files extracted")
        csv_ok = True
    except Exception as e:
        print(f"     ❌ Labels extraction failed: {e}")
else:
    print(f"  ❌ Labels archive not found: {NIPS4B_LABELS_ARCHIVE}")

# ── Update WAV/CSV dirs to extracted paths ────────────────────────────────────
# Override the module-level constants to point at extracted content.
NIPS4B_WAV_DIR = NIPS4B_WAV_EXTRACT   # now writable extracted path
NIPS4B_CSV_DIR = NIPS4B_CSV_EXTRACT   # now writable extracted path

# ── Confirm files found ───────────────────────────────────────────────────────
wav_files = list(NIPS4B_WAV_DIR.rglob("*.wav"))
csv_files = list(NIPS4B_CSV_DIR.rglob("*.csv"))

print()
print(f"  WAV files found:        {len(wav_files)}")
print(f"  Annotation CSVs found:  {len(csv_files)}")
if wav_files:
    print(f"  Sample WAV:             {wav_files[0]}")
if csv_files:
    print(f"  Sample CSV:             {csv_files[0]}")



# ─────────────────────────────────────────────────────────────────────────────
# NIPS4Bplus label format (Kaggle competition dataset)
# ─────────────────────────────────────────────────────────────────────────────
# The Kaggle dataset has GLOBAL CSVs, NOT per-recording CSVs.
# We have these 5 files:
#   nips4b_birdchallenge_espece_list.csv  → species list (ClassId → name)
#   nips4b_birdchallenge_train_labels.csv → main label file (which species in each WAV)
#   numero_file_train.csv                 → file number → filename mapping
#   tps_canaux_sr_nbits_TRAIN.csv         → audio metadata per file
#   example_NIPS4B13_submission_format_test_prediction.csv → submission example (ignore)
#
# Strategy:
#   1. Read espece_list.csv to build ClassId → species name map
#   2. Read train_labels.csv to build {wavfile → [species_present]} map
#   3. For each WAV that has any known species, load audio and extract 4s windows
#   4. Save directly as NPZ (no intermediate cut WAV needed)
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 1: Inspect and load all CSVs ─────────────────────────────────────────
print("\n── Inspecting extracted CSVs ──")
extracted_csvs = sorted(NIPS4B_CSV_EXTRACT.rglob("*.csv"))
csv_by_stem = {p.stem.lower(): p for p in extracted_csvs}
for p in extracted_csvs:
    try:
        tmp = pd.read_csv(p, nrows=2)
        print(f"  {p.name}")
        print(f"    cols: {list(tmp.columns)[:8]}{'...' if len(tmp.columns)>8 else ''}")
        print(f"    row0: {tmp.iloc[0].tolist()[:6] if len(tmp)>0 else '(empty)'}")
    except Exception as e:
        print(f"  {p.name} — could not read: {e}")

# ── Step 2: Find the species list CSV ─────────────────────────────────────────
espece_path = next(
    (p for stem, p in csv_by_stem.items() if "espece" in stem or "species" in stem),
    None
)
# ── Step 3: Find the training labels CSV ──────────────────────────────────────
labels_path = next(
    (p for stem, p in csv_by_stem.items() if "train_label" in stem or "train_labels" in stem),
    None
)
print(f"\n  espece CSV:  {espece_path}")
print(f"  labels CSV:  {labels_path}")

# ── Step 4: Build ClassId → species name map ──────────────────────────────────
classid_to_species = {}   # int → str
if espece_path is not None:
    esp_df = pd.read_csv(espece_path)
    esp_df.columns = [c.strip().lower().replace(" ", "_") for c in esp_df.columns]
    print(f"\n  espece columns: {list(esp_df.columns)}")
    # Try to find class-id and name columns
    id_col   = next((c for c in esp_df.columns if any(k in c for k in ["class", "id", "num", "index"])), None)
    name_col = next((c for c in esp_df.columns if any(k in c for k in ["name", "nom", "species", "espece", "sci", "latin"])), None)
    print(f"  id_col={id_col}  name_col={name_col}")
    if id_col and name_col:
        for _, row in esp_df.iterrows():
            try:
                classid_to_species[int(row[id_col])] = str(row[name_col]).strip()
            except (ValueError, TypeError):
                pass
    elif len(esp_df.columns) >= 2:
        # Fallback: assume first col = id, second col = name
        for _, row in esp_df.iterrows():
            try:
                classid_to_species[int(row.iloc[0])] = str(row.iloc[1]).strip()
            except (ValueError, TypeError):
                pass
    print(f"  Species mapped: {len(classid_to_species)}")
    if classid_to_species:
        sample_items = list(classid_to_species.items())[:5]
        print(f"  Sample: {sample_items}")

# ── Step 5: Build {wavfile_stem → [species]} from train labels CSV ─────────────
file_to_species: dict = {}   # "nips4b_birds_trainfile001" → ["Turdus merula", ...]
if labels_path is not None:
    lbl_df = pd.read_csv(labels_path)
    # Normalise column names
    lbl_df.columns = [c.strip().lstrip("#").lower().replace(" ", "_") for c in lbl_df.columns]
    print(f"\n  labels columns ({len(lbl_df.columns)}): {list(lbl_df.columns)[:10]}")
    print(f"  labels shape: {lbl_df.shape}")

    # Detect file/name column
    file_col = next((c for c in lbl_df.columns if any(k in c for k in ["file", "name", "record", "num"])), None)
    print(f"  file_col={file_col}")

    if file_col is None and len(lbl_df.columns) > 0:
        file_col = lbl_df.columns[0]  # assume first column is filename

    if file_col is not None:
        # Case A: wide format — one row per file, one column per class (multi-hot or prob)
        class_cols = [c for c in lbl_df.columns if c != file_col]
        # If column names look like integers or "class_N", treat as wide
        looks_wide = all(
            c.lstrip("class_").lstrip("prob").replace(".", "").isdigit() or c.isdigit()
            for c in class_cols[:3]
        ) if class_cols else False

        print(f"  Format detection: {'WIDE (multi-hot)' if looks_wide else 'LONG or unknown'}")

        if looks_wide:
            # Wide format: parse class presence from column values
            for _, row in lbl_df.iterrows():
                wav_stem = str(row[file_col]).strip()
                # Remove extension if present
                if "." in wav_stem:
                    wav_stem = wav_stem.rsplit(".", 1)[0]
                species_list = []
                for col in class_cols:
                    try:
                        val = float(row[col])
                        if val > 0:
                            # Try to parse class id from column name
                            col_clean = col.lstrip("class_").lstrip("prob").replace("_", "")
                            class_id  = int(float(col_clean)) if col_clean.replace(".", "").isdigit() else None
                            if class_id is not None and class_id in classid_to_species:
                                species_list.append(classid_to_species[class_id])
                            elif class_id is not None:
                                species_list.append(f"class_{class_id}")
                    except (ValueError, TypeError):
                        pass
                if species_list:
                    file_to_species[wav_stem] = species_list
        else:
            # Long format: one row per (file, class)
            class_col = next((c for c in lbl_df.columns if any(k in c for k in ["class", "id", "label", "species"])), None)
            if class_col is None and len(class_cols) > 0:
                class_col = class_cols[0]
            print(f"  class_col={class_col}")
            if class_col:
                for _, row in lbl_df.iterrows():
                    wav_stem = str(row[file_col]).strip().rsplit(".", 1)[0]
                    try:
                        cid = int(float(row[class_col]))
                        sp  = classid_to_species.get(cid, f"class_{cid}")
                        file_to_species.setdefault(wav_stem, []).append(sp)
                    except (ValueError, TypeError):
                        pass

    print(f"\n  Recordings with labels: {len(file_to_species)}")
    if file_to_species:
        sample_key = next(iter(file_to_species))
        print(f"  Sample: {sample_key} → {file_to_species[sample_key][:3]}")

# ── Step 6: Process WAVs directly → 4s windows → NPZ ─────────────────────────
# Instead of cutting to intermediate WAVs, extract features directly.
# Timestamps are approximated as window start/end (no annotation timestamps
# available from the global format; fine-grained timestamps only exist in
# the figshare per-recording CSVs).
print(f"\nProcessing NIPS4Bplus WAVs → 4s windows → NPZ...")
print(f"  WAV dir: {NIPS4B_WAV_DIR}")
print(f"  Labelled recordings: {len(file_to_species)}")

all_wav_files = list(NIPS4B_WAV_DIR.rglob("*.wav"))
print(f"  Total WAV files: {len(all_wav_files)}")

manifest_rows = []

for wav_path in tqdm(all_wav_files, desc="NIPS4B WAVs"):
    wav_stem = wav_path.stem   # e.g. "nips4b_birds_trainfile013"

    # Get species labels for this recording (may be empty if not in label CSV)
    species_list = file_to_species.get(wav_stem, [])

    try:
        audio, sr = sf.read(str(wav_path))
    except Exception as e:
        log.warning(f"  Cannot read {wav_path.name}: {e}")
        continue

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # Resample to SAMPLE_RATE if different
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # Slice into 4-second non-overlapping windows
    window_samples = int(SAMPLE_RATE * WINDOW_SEC)
    if len(audio) < SAMPLE_RATE:   # < 1 second — skip
        continue

    n_windows = max(1, len(audio) // window_samples)
    for win_idx in range(n_windows):
        t_start = win_idx * WINDOW_SEC
        t_end   = t_start + WINDOW_SEC
        s_start = win_idx * window_samples
        s_end   = s_start + window_samples

        seg = audio[s_start:s_end]
        if len(seg) < window_samples:
            seg = np.pad(seg, (0, window_samples - len(seg)))

        lm = compute_log_mel(seg, sr=SAMPLE_RATE)
        cv = compute_acoustic_concepts(seg, sr=SAMPLE_RATE)

        seg_id   = f"{wav_stem}_win{win_idx:04d}"
        npz_path = NIPS4B_PROC_DIR / f"{seg_id}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            str(npz_path),
            log_mel=lm[np.newaxis, :, :],                          # (1, 128, T)
            concepts=np.array([list(cv.values())], dtype=np.float32),  # (1, 6)
            concept_names=np.array([
                "peak_frequency", "trill_rate", "call_duration",
                "fm_rate", "spectral_centroid", "inter_call_silence"
            ]),
            species_labels=np.array(species_list, dtype=object),    # list of species present
            segment_id=seg_id,
            source_wav=wav_path.name,
            start_sec=np.float64(t_start),   # window start (approximate)
            end_sec=np.float64(t_end),         # window end (approximate)
            duration_sec=np.float64(WINDOW_SEC),
        )

        manifest_rows.append({
            "segment_id"  : seg_id,
            "source_wav"  : wav_path.name,
            "species"     : "|".join(species_list) if species_list else "unlabelled",
            "start_sec"   : t_start,
            "end_sec"     : t_end,
            "duration_sec": WINDOW_SEC,
            "sample_rate" : SAMPLE_RATE,
            "output_path" : str(npz_path),
        })

nips4b_manifest = pd.DataFrame(manifest_rows)

# ── Save manifest ──────────────────────────────────────────────────────────────
nips4b_manifest_path = PROC_DIR / "nips4bplus_cut_manifest.csv"
nips4b_manifest.to_csv(nips4b_manifest_path, index=False)

print(f"\nNIPS4Bplus processing complete:")
print(f"  Total 4s windows:   {len(nips4b_manifest)}")
if len(nips4b_manifest) > 0 and "species" in nips4b_manifest.columns:
    labelled = nips4b_manifest[nips4b_manifest["species"] != "unlabelled"]
    print(f"  Labelled windows:   {len(labelled)}")
    print(f"  Unlabelled windows: {len(nips4b_manifest) - len(labelled)}")
    print(f"  Manifest saved → {nips4b_manifest_path}")
    if len(nips4b_manifest) > 0:
        print(f"\nSample rows:")
        print(nips4b_manifest.head(5).to_string())
else:
    print("  ⚠️  No windows extracted. Check WAV files and labels.")
    print(f"  file_to_species entries: {len(file_to_species)}")
    print(f"  WAV files found:         {len(all_wav_files)}")




# ─────────────────────────────────────────────
# CELL 9 — NIPS4Bplus feature status check
#           Features are already written to NPZ in Cell 8.
#           This cell just counts and confirms output.
# ─────────────────────────────────────────────
# %%

npz_files_nips4b = list(NIPS4B_PROC_DIR.rglob("*.npz"))
print(f"✅ NIPS4Bplus NPZ files written: {len(npz_files_nips4b)}")
print(f"   Location: {NIPS4B_PROC_DIR}")

if len(npz_files_nips4b) == 0:
    print("   ⚠️  No NPZ files found — check Cell 8 output above.")
else:
    # Spot-check one NPZ
    sample_npz = np.load(str(npz_files_nips4b[0]), allow_pickle=True)
    print(f"   Sample NPZ keys:      {list(sample_npz.keys())}")
    print(f"   log_mel shape:        {sample_npz['log_mel'].shape}")
    print(f"   concepts shape:       {sample_npz['concepts'].shape}")
    print(f"   species_labels:       {sample_npz['species_labels'].tolist()}")
    print(f"   start_sec:            {float(sample_npz['start_sec']):.1f}s")
    print(f"   end_sec:              {float(sample_npz['end_sec']):.1f}s")

# Write a features manifest for downstream cells to reference
nips4b_feat_manifest_path = PROC_DIR / "nips4bplus_features_manifest.csv"
nips4b_manifest.to_csv(nips4b_feat_manifest_path, index=False)
print(f"   Features manifest → {nips4b_feat_manifest_path}")


# ─────────────────────────────────────────────
# CELL 10 — Species overlap, metadata JSONs & disk audit
# ─────────────────────────────────────────────
# %%

# ── Species overlap: BirdCLEF training ∩ NIPS4Bplus validation ──────────────
validation_overlap_species = sorted([
    s for s in accepted_species
    if s in NIPS4B_MODERN_NAMES
])

print(f"{'='*60}")
print(f"SPECIES OVERLAP SUMMARY")
print(f"{'='*60}")
print(f"  BirdCLEF accepted species:     {len(accepted_species)}")
print(f"  NIPS4Bplus target species:     {len(NIPS4BPLUS_SPECIES)}")
print(f"  Validation overlap species:    {len(validation_overlap_species)}")

missing_from_training = [s for s in NIPS4BPLUS_SPECIES if s not in accepted_species]
if missing_from_training:
    print(f"\n  NIPS4Bplus species NOT found in BirdCLEF 2021 accepted set:")
    for s in missing_from_training:
        print(f"    - {s}")

# ── Save JSON metadata files ─────────────────────────────────
PROC_DIR.mkdir(parents=True, exist_ok=True)

train_list_path = PROC_DIR / "species_list_train.json"
val_list_path   = PROC_DIR / "species_list_validation_overlap.json"
label_map_path  = PROC_DIR / "label_map.json"

with open(train_list_path, "w") as f:
    json.dump({
        "species": accepted_species,
        "count"  : len(accepted_species),
        "source" : "BirdCLEF 2021 (filtered, rating >= 3.5, cap 50/species)",
        "note"   : "Accepted BirdCLEF 2021 species with sufficient high-quality clips.",
    }, f, indent=2)

with open(val_list_path, "w") as f:
    json.dump({
        "species": validation_overlap_species,
        "count"  : len(validation_overlap_species),
        "note"   : (
            "Subset of training species appearing in NIPS4Bplus. "
            "Used for FAITHFULNESS EVALUATION ONLY. "
            "Never mix with the training species scope."
        ),
    }, f, indent=2)

# Rebuild label map from sorted accepted species
label2idx_sorted = {sp: i for i, sp in enumerate(sorted(accepted_species))}
idx2label_sorted = {str(i): sp for sp, i in label2idx_sorted.items()}

with open(label_map_path, "w") as f:
    json.dump({"label2idx": label2idx_sorted, "idx2label": idx2label_sorted}, f, indent=2)

print(f"\nJSON metadata files saved:")
print(f"  → species_list_train.json              ({len(accepted_species)} species)")
print(f"  → species_list_validation_overlap.json ({len(validation_overlap_species)} species)")
print(f"  → label_map.json                       ({len(label2idx_sorted)} classes)")

# ── Disk audit ───────────────────────────────────────────────
usage    = shutil.disk_usage("/kaggle/working")
used_gb  = usage.used / 1e9
total_gb = usage.total / 1e9
print(f"\n{'='*60}")
print(f"DISK USAGE AUDIT")
print(f"{'='*60}")
print(f"  /kaggle/working used:  {used_gb:.2f} GB / {total_gb:.2f} GB")
print(f"  Limit guard:           < {MAX_DISK_GB} GB")

if used_gb > MAX_DISK_GB:
    print(f"\n  ⚠️  WARNING: disk usage ({used_gb:.2f} GB) exceeds soft limit ({MAX_DISK_GB} GB)!")
    print("     Review largest files and consider reducing MAX_CLIPS_PER_SP.")
else:
    print(f"  ✅ Disk usage is within the {MAX_DISK_GB} GB soft limit.")


# ─────────────────────────────────────────────
# CELL 11 — Phase 1 Final Summary & Checklist
# ─────────────────────────────────────────────
# %%

def check_item(label: str, path_or_cond):
    if isinstance(path_or_cond, bool):
        status = "OK" if path_or_cond else "MISSING"
        note   = ""
    else:
        p    = Path(path_or_cond)
        ok   = p.exists() and p.stat().st_size > 0
        status = "OK" if ok else "MISSING"
        note = f"({p.stat().st_size // 1024} KB)" if ok else "(file missing or empty)"
    symbol = "[OK]" if status == "OK" else "[!!]"
    print(f"  {symbol}  {label:55s} {note}")

print("=" * 70)
print("PHASE 1 COMPLETION CHECKLIST")
print("=" * 70)
check_item("birdclef_manifest.csv",                 PROC_DIR / "birdclef_manifest.csv")
check_item("birdclef_extraction_log.csv",           PROC_DIR / "birdclef_extraction_log.csv")
check_item("nips4bplus_cut_manifest.csv",            PROC_DIR / "nips4bplus_cut_manifest.csv")
check_item("nips4bplus_features_manifest.csv",       PROC_DIR / "nips4bplus_features_manifest.csv")
check_item("species_list_train.json",               train_list_path)
check_item("species_list_validation_overlap.json",  val_list_path)
check_item("label_map.json",                        label_map_path)
check_item("BirdCLEF train NPZ dir non-empty",      len(list((BIRDCLEF_PROC_DIR / "train").glob("*.npz"))) > 0)
check_item("NIPS4Bplus validation NPZ dir non-empty", len(list(NIPS4B_PROC_DIR.glob("*.npz"))) > 0)

print()
print(f"  BirdCLEF species accepted:     {len(accepted_species)}")
print(f"  Validation overlap species:    {len(validation_overlap_species)}")

bc_ok = bc_log_df[bc_log_df["status"] == "ok"]
print(f"  BirdCLEF segments extracted:   {int(bc_ok['segments_saved'].sum())}")
print(f"  Disk used (working):           {used_gb:.2f} GB")

print()
print("NEXT STEPS:")
print("  1. 'Save Version' this notebook → creates the output dataset.")
print("  2. Rename output dataset to 'bird-xai-processed-features'.")
print("  3. Attach 'bird-xai-processed-features' dataset to Notebook 2 (training).")
print("  4. Run Notebook 2 to train the CBM and black-box baseline.")
print()
print("IMPORTANT: DO NOT delete /kaggle/working/processed_features/ before saving.")
