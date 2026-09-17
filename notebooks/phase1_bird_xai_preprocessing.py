# =============================================================================
# ORNITHO-EX | Notebook 1: bird-xai-preprocessing
# Phase 1 — Dataset Acquisition (Xeno-canto + NIPS4Bplus)
# =============================================================================
# IMPORTANT: Before running this notebook:
#   1. Enable Internet: Settings > Internet > ON
#   2. GPU accelerator: OFF (this notebook is CPU-only)
#   3. Runtime: Run all cells top-to-bottom
# =============================================================================

# ─────────────────────────────────────────────
# CELL 1 — Install dependencies
# ─────────────────────────────────────────────
# %%
import subprocess, sys

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

pip_install(
    "librosa==0.10.2",
    "soundfile",
    "tqdm",
    "pandas",
    "requests",
    "numpy",
)

print("✅ Dependencies installed.")


# ─────────────────────────────────────────────
# CELL 2 — Imports
# ─────────────────────────────────────────────
# %%
import os
import json
import time
import random
import logging
import hashlib
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ornitho-ex")

print("✅ Imports done.")


# ─────────────────────────────────────────────
# CELL 3 — Configuration & Species Lists
# ─────────────────────────────────────────────
# %%

# ── Paths (Kaggle environment) ──────────────────────────────
BASE_OUT       = Path("/kaggle/working")
RAW_XC_DIR     = BASE_OUT / "raw_audio" / "xenocanto"
RAW_NIPS_DIR   = BASE_OUT / "raw_audio" / "nips4bplus"
PROC_XC_DIR    = BASE_OUT / "xeno_canto"
PROC_NIPS_DIR  = BASE_OUT / "nips4bplus" / "validation"

# Create all directories upfront
for d in [RAW_XC_DIR, RAW_NIPS_DIR, PROC_XC_DIR / "train",
          PROC_XC_DIR / "val", PROC_XC_DIR / "test", PROC_NIPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Xeno-canto API settings (API v3) ────────────────────────
XC_API_BASE      = "https://xeno-canto.org/api/3/recordings"

# Kaggle Secrets: To use the API, add a secret named "XC_API_KEY" with your key
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    XC_API_KEY = user_secrets.get_secret("XC_API_KEY")
    print("✅ Loaded XC_API_KEY from Kaggle Secrets")
except Exception:
    print("⚠️  XC_API_KEY not found in Kaggle Secrets! Please set it.")
    XC_API_KEY = "YOUR_API_KEY_HERE"

QUALITY_FILTER   = ["A", "B"]
MIN_CLIPS        = 80     # drop species below this (insufficient data)
MAX_CLIPS        = 100    # REDUCED to 100 to stay under Kaggle's 20GB limit
MAX_RETRIES      = 5      # API retry attempts
BACKOFF_BASE     = 2      # seconds — exponential backoff base
MAX_LENGTH_SEC   = 120    # Skip clips longer than 2 minutes to save disk space

# ── Audio / Feature settings ─────────────────────────────────
TARGET_SR        = 44100  # matches NIPS4Bplus
SEGMENT_SEC      = 4      # seconds per window
N_MELS           = 128
N_FFT            = 2048
HOP_LENGTH       = 512

# ── Species lists ────────────────────────────────────────────
# Core 51 European species from NIPS4Bplus
# (these are the species for which faithfulness evaluation is possible)
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

# Extended species — additional Xeno-canto species for broader training scope
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

# Full training list
TRAINING_SPECIES = NIPS4BPLUS_SPECIES + EXTENDED_SPECIES

print(f"✅ Config ready.")
print(f"   Training species: {len(TRAINING_SPECIES)}")
print(f"   NIPS4Bplus species (validation overlap): {len(NIPS4BPLUS_SPECIES)}")
print(f"   Extended species: {len(EXTENDED_SPECIES)}")
print(f"   Quality filter: {QUALITY_FILTER}")
print(f"   Min clips/species: {MIN_CLIPS}  |  Max clips/species: {MAX_CLIPS}")


# ─────────────────────────────────────────────
# CELL 4 — Xeno-canto API helpers (retry + backoff)
# ─────────────────────────────────────────────
# %%

# ── Taxonomy translations (2013 -> Modern) ───────────────────
# NIPS4Bplus uses 2013 taxonomy. Xeno-canto API requires modern taxonomy.
TAXONOMY_MAP = {
    "Carduelis chloris": "Chloris chloris",
    "Carduelis spinus": "Spinus spinus",
    "Parus ater": "Periparus ater",
    "Parus caeruleus": "Cyanistes caeruleus",
    "Parus palustris": "Poecile palustris",
    "Saxicola torquata": "Saxicola rubicola",
    "Sylvia communis": "Curruca communis",
    "Sylvia curruca": "Curruca curruca",
    "Sylvia nisoria": "Curruca nisoria",
}

def xc_api_query(species_name: str, quality: str, page: int = 1) -> dict:
    """
    Query the Xeno-canto API for recordings of a species.
    Applies exponential backoff on failure.
    Returns the parsed JSON response dict, or an empty dict on failure.
    """
    # Translate to modern taxonomy if needed for the API call
    query_name = TAXONOMY_MAP.get(species_name, species_name)
    
    # Extract Genus and Species for precise API v3 search tags
    parts = query_name.split(" ", 1)
    genus = parts[0]
    species = parts[1] if len(parts) > 1 else ""
    
    # API v3 strict syntax: gen:Genus sp:species q:A
    # No quotes around values, no OR operators
    query_str = f"gen:{genus} sp:{species} q:{quality}" if species else f"gen:{genus} q:{quality}"
    
    params = {
        "query": query_str,
        "page": page,
        "key": XC_API_KEY
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(XC_API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            wait = BACKOFF_BASE ** attempt + random.uniform(0, 1)
            log.warning(
                f"[{species_name}] API attempt {attempt}/{MAX_RETRIES} failed: {e}. "
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)

    log.error(f"[{species_name}] All {MAX_RETRIES} API attempts failed. Skipping.")
    return {}


def parse_length(length_str: str) -> int:
    """Convert '1:23' to 83 seconds. Return 0 if invalid."""
    try:
        if ":" in length_str:
            parts = length_str.split(":")
            return int(parts[0]) * 60 + int(parts[1])
        return int(length_str)
    except (ValueError, TypeError):
        return 0

def fetch_all_recordings(species_name: str) -> list:
    """
    Paginate through Xeno-canto results for a species.
    Fetches quality 'A' first, then 'B' if needed to reach MIN_CLIPS.
    Caps at MAX_CLIPS to keep the dataset balanced.
    Ignores clips longer than MAX_LENGTH_SEC to prevent disk exhaustion.
    """
    all_recordings = []

    for q in QUALITY_FILTER:
        # Fetch first page for this quality
        first = xc_api_query(species_name, quality=q, page=1)
        if not first or "recordings" not in first:
            continue
            
        num_pages = int(first.get("numPages", 1))
        
        # Filter page 1 by length
        valid_clips = [r for r in first["recordings"] if 0 < parse_length(r.get("length", "0")) <= MAX_LENGTH_SEC]
        all_recordings.extend(valid_clips)
        
        # Stop early if we already hit MAX_CLIPS for this species just from page 1
        if len(all_recordings) >= MAX_CLIPS:
            break

        # Fetch remaining pages for this quality
        for page in range(2, num_pages + 1):
            data = xc_api_query(species_name, quality=q, page=page)
            if data and "recordings" in data:
                valid_clips = [r for r in data["recordings"] if 0 < parse_length(r.get("length", "0")) <= MAX_LENGTH_SEC]
                all_recordings.extend(valid_clips)
            time.sleep(0.3)  # be polite to the API
            
            # Stop paginating if we have enough
            if len(all_recordings) >= MAX_CLIPS:
                break
                
        if len(all_recordings) >= MAX_CLIPS:
            break

    # Strictly filter to A/B and cap at MAX_CLIPS
    filtered = [r for r in all_recordings if r.get("q", "") in QUALITY_FILTER]
    return filtered[:MAX_CLIPS]


def safe_filename(species_name: str) -> str:
    """Convert 'Turdus merula' to 'Turdus_merula'"""
    return species_name.replace(" ", "_")


print("✅ API helper functions defined.")


# ─────────────────────────────────────────────
# CELL 5 — Phase 1A: Query Xeno-canto API for all species
#           (Only queries metadata, does NOT download audio yet)
# ─────────────────────────────────────────────
# %%

# Check internet connectivity
try:
    ping = requests.get("https://xeno-canto.org", timeout=10)
    print(f"✅ Internet OK — xeno-canto.org reachable (status {ping.status_code})")
except Exception as e:
    print(f"❌ Internet NOT reachable: {e}")
    print("   Go to: Notebook Settings > Internet > ON, then restart the kernel.")
    raise SystemExit("Internet required for Phase 1A.")

# ── Query all species, collect metadata ─────────────────────
species_metadata = {}   # {species_name: [recording_dict, ...]}
skipped_species  = []   # species with fewer than MIN_CLIPS clips

print(f"\nQuerying Xeno-canto API for {len(TRAINING_SPECIES)} species...\n")

for species in tqdm(TRAINING_SPECIES, desc="Species"):
    recordings = fetch_all_recordings(species)
    count = len(recordings)

    if count < MIN_CLIPS:
        log.warning(f"  {species}: only {count} A/B clips — BELOW threshold ({MIN_CLIPS}), skipping.")
        skipped_species.append({"species": species, "available_clips": count, "reason": f"< {MIN_CLIPS} clips"})
        continue

    # Cap to MAX_CLIPS (prefer A-quality, then B)
    a_clips = [r for r in recordings if r.get("q") == "A"]
    b_clips = [r for r in recordings if r.get("q") == "B"]
    selected = (a_clips + b_clips)[:MAX_CLIPS]

    species_metadata[species] = selected
    log.info(f"  {species}: {count} available -> keeping {len(selected)} (A:{len(a_clips)}, B:{len(b_clips)})")
    time.sleep(0.3)  # rate-limit courtesy pause

# ── Summary ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PHASE 1A QUERY SUMMARY")
print(f"{'='*60}")
print(f"  Species queried:     {len(TRAINING_SPECIES)}")
print(f"  Species accepted:    {len(species_metadata)}")
print(f"  Species skipped:     {len(skipped_species)}")
print(f"  Total clips to DL:   {sum(len(v) for v in species_metadata.values())}")

if skipped_species:
    print(f"\n  Skipped species (< {MIN_CLIPS} clips):")
    for s in skipped_species:
        print(f"    - {s['species']}: {s['available_clips']} clips")

# ── Save query results for inspection ────────────────────────
QUERY_RESULTS_PATH = BASE_OUT / "xc_query_results.json"
with open(QUERY_RESULTS_PATH, "w") as f:
    json.dump({
        "accepted": {
            k: [{"id": r["id"], "q": r["q"], "file-name": r.get("file-name", ""),
                 "length": r.get("length", ""), "type": r.get("type", ""),
                 "file": r.get("file", "")}
                for r in v]
            for k, v in species_metadata.items()
        },
        "skipped": skipped_species,
    }, f, indent=2)

print(f"\n✅ Query results saved -> {QUERY_RESULTS_PATH}")
print("   Review the skipped species above before proceeding to the download cell.")


# ─────────────────────────────────────────────
# CELL 6 — Phase 1A: Download Xeno-canto Audio
# ─────────────────────────────────────────────
# %%

def download_file(url: str, dest_path: Path, retries: int = MAX_RETRIES) -> bool:
    """
    Download a file from url to dest_path.
    Returns True on success, False on failure after all retries.
    Idempotent: skips if file already exists and is non-empty.
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True  # already downloaded

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True
        except requests.RequestException as e:
            wait = BACKOFF_BASE ** attempt + random.uniform(0, 1)
            log.warning(f"    Attempt {attempt}/{retries} failed: {e}. Retry in {wait:.1f}s...")
            # Remove partial file before retrying
            if dest_path.exists():
                dest_path.unlink()
            time.sleep(wait)

    log.error(f"    FAILED after {retries} attempts: {url}")
    return False


# ── Download loop ─────────────────────────────────────────────
download_log = []
total_clips = sum(len(recs) for recs in species_metadata.values())

print(f"Downloading {total_clips} audio clips to {RAW_XC_DIR} ...")
print("(Idempotent — safe to re-run if interrupted)\n")

with tqdm(total=total_clips, desc="Downloading", unit="clip") as pbar:
    for species, recordings in species_metadata.items():
        sp_dir = RAW_XC_DIR / safe_filename(species)
        sp_dir.mkdir(parents=True, exist_ok=True)

        for rec in recordings:
            rec_id   = rec.get("id", "unknown")
            quality  = rec.get("q", "?")
            filename = rec.get("file-name", f"{rec_id}.mp3")
            dl_url   = rec.get("file", "")

            if not dl_url:
                log.warning(f"No download URL for rec {rec_id} of {species}, skipping.")
                pbar.update(1)
                continue

            # Sanitize filename
            filename = filename if filename else f"xc{rec_id}.mp3"
            dest = sp_dir / filename

            success = download_file(dl_url, dest)

            download_log.append({
                "species":       species,
                "rec_id":        rec_id,
                "quality":       quality,
                "filename":      filename,
                "local_path":    str(dest),
                "length_sec":    rec.get("length", ""),
                "rec_type":      rec.get("type", ""),
                "country":       rec.get("cnt", ""),
                "download_ok":   success,
            })
            pbar.update(1)

# ── Save download manifest ─────────────────────────────────────
manifest_df = pd.DataFrame(download_log)
manifest_path = BASE_OUT / "xc_download_manifest.csv"
manifest_df.to_csv(manifest_path, index=False)

successful = manifest_df["download_ok"].sum()
failed     = (~manifest_df["download_ok"]).sum()

print(f"\n{'='*60}")
print(f"PHASE 1A DOWNLOAD SUMMARY")
print(f"{'='*60}")
print(f"  Total clips attempted: {len(download_log)}")
print(f"  Successful:            {successful}")
print(f"  Failed:                {failed}")
print(f"  Manifest saved ->      {manifest_path}")

if failed > 0:
    print(f"\nFailed downloads (re-run this cell to retry — it's idempotent):")
    fail_df = manifest_df[~manifest_df["download_ok"]]
    print(fail_df[["species", "rec_id", "quality"]].to_string(index=False))


# ─────────────────────────────────────────────
# CELL 7 — Phase 1B: Download NIPS4Bplus Dataset
#           Sources confirmed from GitHub: github.com/fbravosanchez/NIPS4Bplus
# ─────────────────────────────────────────────
# %%
import os
# Path to the extracted WAV files (check with: os.listdir("/kaggle/input/nips4b-audio-raw/"))
NIPS4B_WAV_DIR  = Path("/kaggle/input/nips4b-audio-raw")   # adjust subfolder if needed
# Path to the annotation CSVs (check with: os.listdir("/kaggle/input/nips4bplus-annotations/"))
NIPS4B_CSV_DIR  = Path("/kaggle/input/nips4bplus-annotations")  # adjust subfolder if needed
# Quick sanity check
wav_files = list(NIPS4B_WAV_DIR.rglob("*.wav"))
csv_files = list(NIPS4B_CSV_DIR.rglob("*.csv"))
print(f"WAV files found:        {len(wav_files)}")
print(f"Annotation CSVs found:  {len(csv_files)}")
if len(wav_files) == 0:
    print("  -> Check the path. Run: os.listdir('/kaggle/input/nips4b-audio-raw/')")
if len(csv_files) == 0:
    print("  -> Check the path. Run: os.listdir('/kaggle/input/nips4bplus-annotations/')")
# Output dirs (writable)
NIPS4B_CUT_DIR   = BASE_OUT / "nips4bplus" / "cut_wavs"
NIPS4B_CUT_DIR.mkdir(parents=True, exist_ok=True)
print("\nPaths set. Proceed to Cell 8 to run the cutting script.")

# ─────────────────────────────────────────────
# CELL 8 — Phase 1B: Cut NIPS4Bplus Files
#           Implements the logic of cut_nips4bplus_files.py inline
#           (preserves all timestamp metadata — NEVER discard)
# ─────────────────────────────────────────────
# %%
import soundfile as sf

def cut_nips4bplus(wav_dir: Path, csv_dir: Path, output_dir: Path) -> pd.DataFrame:
    """
    Port of cut_nips4bplus_files.py from github.com/fbravosanchez/NIPS4Bplus.
    Reads each per-recording CSV annotation from NIPS4Bplus and cuts the
    corresponding WAV file into short segments based on the tag timestamps.

    CRITICAL: timestamp metadata is preserved in the returned manifest.
    These timestamps are required for faithfulness evaluation against Grad-CAM
    and deletion/insertion AUC (per project guidelines §8.6).

    Args:
        wav_dir:    folder containing original NIPS4B .wav files
        csv_dir:    folder containing NIPS4Bplus annotation .csv files
                    (one CSV per recording, matching filename prefix)
        output_dir: where to write the cut .wav segments
    Returns:
        DataFrame manifest with columns:
          segment_id, source_wav, species, start_sec, end_sec,
          duration_sec, output_path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all annotation CSVs
    csv_files = sorted(csv_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    manifest_rows = []
    skipped = 0

    for csv_path in tqdm(csv_files, desc="Cutting NIPS4B files"):
        # Annotation CSV name encodes the source wav name:
        # e.g. "nips4b_birds_trainfile001.csv" → "nips4b_birds_trainfile001.wav"
        wav_stem = csv_path.stem  # filename without extension
        wav_file = wav_dir / f"{wav_stem}.wav"

        # Also try in subdirectories
        if not wav_file.exists():
            matches = list(wav_dir.rglob(f"{wav_stem}.wav"))
            wav_file = matches[0] if matches else wav_file

        if not wav_file.exists():
            log.warning(f"  WAV not found for {csv_path.name}: {wav_file}")
            skipped += 1
            continue

        # Read annotation CSV
        # Expected columns: Starttime, Stoptime, (species/label column)
        try:
            ann = pd.read_csv(csv_path)
        except Exception as e:
            log.warning(f"  Could not read {csv_path.name}: {e}")
            skipped += 1
            continue

        # Normalize column names (may vary slightly across files)
        ann.columns = [c.strip().lower().replace(" ", "_") for c in ann.columns]

        # Identify time columns (robust to different naming conventions)
        start_col = next((c for c in ann.columns if "start" in c), None)
        stop_col  = next((c for c in ann.columns if "stop" in c or "end" in c), None)
        # Identify species/label column
        label_col = next((c for c in ann.columns
                          if any(k in c for k in ["species", "label", "class", "espece"])), None)

        if start_col is None or stop_col is None:
            log.warning(f"  Cannot find time columns in {csv_path.name}: {list(ann.columns)}")
            skipped += 1
            continue

        # Read audio file once per recording
        try:
            audio, sr = sf.read(str(wav_file))
        except Exception as e:
            log.warning(f"  Cannot read audio {wav_file}: {e}")
            skipped += 1
            continue

        # Convert stereo to mono if needed
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Cut one segment per annotation row
        for row_idx, row in ann.iterrows():
            try:
                t_start = float(row[start_col])
                t_end   = float(row[stop_col])
            except (ValueError, KeyError):
                continue

            if t_end <= t_start:
                continue  # invalid annotation

            # Sample-level indices
            s_start = int(t_start * sr)
            s_end   = int(t_end   * sr)
            segment = audio[s_start:s_end]

            if len(segment) == 0:
                continue

            # Species label (may be None if not annotated)
            species = str(row[label_col]).strip() if label_col else "unknown"
            if species.lower() in ("nan", "none", ""):
                species = "unknown"

            # Output filename encodes all metadata
            seg_id  = f"{wav_stem}_seg{row_idx:04d}"
            out_name = f"{seg_id}.wav"
            out_path = output_dir / species.replace(" ", "_") / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            sf.write(str(out_path), segment, sr)

            manifest_rows.append({
                "segment_id":    seg_id,
                "source_wav":    wav_file.name,
                "species":       species,
                "start_sec":     t_start,
                "end_sec":       t_end,
                "duration_sec":  round(t_end - t_start, 4),
                "sample_rate":   sr,
                "output_path":   str(out_path),
            })

    manifest = pd.DataFrame(manifest_rows)
    log.info(f"Cutting complete: {len(manifest)} segments, {skipped} files skipped.")
    return manifest


# ── Check prerequisites ───────────────────────────────────────
wav_files = list(NIPS4B_WAV_DIR.rglob("*.wav"))
csv_files = list(NIPS4B_CSV_DIR.rglob("*.csv"))

print(f"WAV files found: {len(wav_files)}")
print(f"Annotation CSVs found: {len(csv_files)}")

if len(wav_files) == 0 or len(csv_files) == 0:
    print("\nPrerequisites not met. Complete Cell 7 first:")
    print("  WAV files needed: download NIPS4B audio")
    print("  CSV files needed: download NIPS4Bplus annotations")
else:
    print(f"\nRunning cut_nips4bplus logic...")
    nips4b_manifest = cut_nips4bplus(
        wav_dir    = NIPS4B_WAV_DIR,
        csv_dir    = NIPS4B_CSV_DIR,
        output_dir = NIPS4B_CUT_DIR,
    )

    # Save manifest — TIMESTAMP METADATA PRESERVED (never discard)
    nips4b_manifest_path = RAW_NIPS_DIR / "nips4bplus_cut_manifest.csv"
    nips4b_manifest.to_csv(nips4b_manifest_path, index=False)

    print(f"\nNIPS4Bplus cutting complete:")
    print(f"  Total segments cut:   {len(nips4b_manifest)}")
    print(f"  Unique species:       {nips4b_manifest['species'].nunique()}")
    print(f"  Manifest saved ->     {nips4b_manifest_path}")
    print(f"\nSample rows:")
    print(nips4b_manifest.head(8).to_string())


# ─────────────────────────────────────────────
# CELL 9 — Compute Species Overlap & Save Lists
# ─────────────────────────────────────────────
# %%

# Accepted training species (those that passed MIN_CLIPS threshold)
accepted_training_species = sorted(list(species_metadata.keys()))

# Validation overlap = accepted training species that are also in NIPS4Bplus
validation_overlap_species = sorted([
    s for s in accepted_training_species
    if s in NIPS4BPLUS_SPECIES
])

print(f"{'='*60}")
print(f"SPECIES OVERLAP SUMMARY")
print(f"{'='*60}")
print(f"  Accepted training species:     {len(accepted_training_species)}")
print(f"  NIPS4Bplus target species:     {len(NIPS4BPLUS_SPECIES)}")
print(f"  Validation overlap species:    {len(validation_overlap_species)}")

missing_from_training = [s for s in NIPS4BPLUS_SPECIES if s not in accepted_training_species]
if missing_from_training:
    print(f"\n  NIPS4Bplus species missing from training (not enough XC data):")
    for s in missing_from_training:
        print(f"    - {s}")

# ── Save species list JSONs ───────────────────────────────────
train_list_path = BASE_OUT / "species_list_train.json"
val_list_path   = BASE_OUT / "species_list_validation_overlap.json"

with open(train_list_path, "w") as f:
    json.dump({
        "species": accepted_training_species,
        "count": len(accepted_training_species),
        "note": "Species with >= MIN_CLIPS A/B quality recordings on Xeno-canto. Used for model training.",
    }, f, indent=2)

with open(val_list_path, "w") as f:
    json.dump({
        "species": validation_overlap_species,
        "count": len(validation_overlap_species),
        "note": "Subset of training species that appear in NIPS4Bplus. Used for FAITHFULNESS EVALUATION ONLY. Never mix with training species scope.",
    }, f, indent=2)

# ── Class label mapping ───────────────────────────────────────
# Sorted alphabetically for reproducibility
label2idx = {sp: i for i, sp in enumerate(accepted_training_species)}
idx2label = {str(i): sp for sp, i in label2idx.items()}

label_map_path = BASE_OUT / "label_map.json"
with open(label_map_path, "w") as f:
    json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, indent=2)

print(f"\nFiles saved:")
print(f"  -> species_list_train.json              ({len(accepted_training_species)} species)")
print(f"  -> species_list_validation_overlap.json ({len(validation_overlap_species)} species)")
print(f"  -> label_map.json                       ({len(label2idx)} classes)")


# ─────────────────────────────────────────────
# CELL 10 — Phase 1 Final Summary & Checklist
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
    print(f"  {symbol}  {label:50s} {note}")

print("=" * 60)
print("PHASE 1 COMPLETION CHECKLIST")
print("=" * 60)

check_item("xc_query_results.json",              BASE_OUT / "xc_query_results.json")
check_item("xc_download_manifest.csv",           BASE_OUT / "xc_download_manifest.csv")
check_item("nips4bplus annotations.csv",         RAW_NIPS_DIR / "annotations.csv")
check_item("nips4bplus species_list.csv",        RAW_NIPS_DIR / "species_list.csv")
check_item("species_list_train.json",            train_list_path)
check_item("species_list_validation_overlap.json", val_list_path)
check_item("label_map.json",                     label_map_path)
check_item("NIPS4B audio dir set",               NIPS4B_AUDIO_INPUT_DIR is not None)

print()
ok_dl  = manifest_df["download_ok"].sum()
bad_dl = (~manifest_df["download_ok"]).sum()
print(f"  Xeno-canto audio downloaded:   {ok_dl} OK  /  {bad_dl} failed")
print(f"  Training species accepted:     {len(accepted_training_species)}")
print(f"  Validation overlap species:    {len(validation_overlap_species)}")

print()
print("NEXT STEPS:")
print("  1. If any XC downloads failed, re-run Cell 6 (safe to re-run).")
print("  2. Download NIPS4B audio from Zenodo (see Cell 7 instructions).")
print("  3. Upload NIPS4B audio as Kaggle Dataset 'nips4b-audio-raw'.")
print("  4. Set NIPS4B_AUDIO_INPUT_DIR in Cell 7 and re-run Cells 7-8.")
print("  5. Save this notebook version before moving to Phase 2.")
print()
print("After saving: outputs become the 'bird-xai-processed-features' dataset.")
print("Attach it to the next notebook before running Phase 2.")
