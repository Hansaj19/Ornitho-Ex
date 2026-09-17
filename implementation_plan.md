# Implementation Plan: Switch to Kaggle BirdCLEF 2021 Dataset

## Goal Description

Pivot Notebook 1 (`bird-xai-preprocessing`) from live Xeno-canto REST API downloads to the pre-hosted Kaggle **BirdCLEF 2021** competition dataset (`/kaggle/input/birdclef-2021`).

This completely eliminates:
1. Live network downloads and API rate-limiting/timeouts.
2. Raw audio disk consumption in `/kaggle/working/` (the 30+ GB of audio remains in read-only `/kaggle/input/`, using **0 MB** of the 20 GB quota).
3. The `OSError: [Errno 28] No space left on device` crash.

The notebook will read short audio files directly from `/kaggle/input/birdclef-2021/train_short_audio/`, extract 4-second segments, log-mel spectrograms (`np.float16`), and the 6 programmatic concept targets on-the-fly, writing only compact feature files (~500 MB) to `/kaggle/working/processed_features/`.

---

## User Review Required

> [!IMPORTANT]
> **Action Required in Kaggle Notebook:**
> Before running the updated notebook, you will need to attach the **BirdCLEF 2021** dataset to your notebook in Kaggle:
> 1. Open your `bird-xai-preprocessing` notebook in Kaggle.
> 2. On the right-side panel, click **"+ Add Data"** (or **"Add Input"**).
> 3. In the search box, type: `birdclef-2021` (from the competition *BirdCLEF 2021 - Birdcall Identification*).
> 4. Click **"Add"**. It will mount at `/kaggle/input/birdclef-2021/`.

> [!NOTE]
> **Taxonomy & Species Selection:**
> BirdCLEF 2021 contains 397 bird species with high-quality focal recordings sourced from Xeno-canto.
> We will filter `train_metadata.csv` to:
> - Keep species matching our target scientific names (including NIPS4Bplus European species present in BirdCLEF 2021).
> - Select high-quality recordings (`rating >= 3.5`).
> - Cap clips per species (e.g. 40–50 clips) to maintain balanced classes.

---

## Open Questions

> [!NOTE]
> 1. **Training Species Count:** BirdCLEF 2021 has 397 species. Would you prefer to train on:
>    - **(A) Targeted European Focus (Recommended):** All species that overlap with European / NIPS4Bplus species + extended European birds in BirdCLEF (~50–60 species), OR
>    - **(B) Top-N Most Common Species:** The top 50–70 most frequent species in BirdCLEF 2021 with the highest recording quality?
>    *(Defaulting to (A) ensures maximum overlap with NIPS4Bplus for faithfulness evaluation).*

---

## Proposed Changes

```
/kaggle/input/birdclef-2021/ (Read-only, 0 MB disk used in working dir)
   ├── train_metadata.csv
   └── train_short_audio/<species_code>/*.ogg
            │
            ▼
Filtered by Species & Quality (rating >= 3.5)
            │
            ▼
On-the-fly Audio Slicing (4.0s windows) & Feature Extraction
   ├── Log-mel Spectrograms (128 mels × 345 frames, float16)
   └── 6 Programmatic Concept Targets (peak freq, trill rate, etc.)
            │
            ▼
/kaggle/working/processed_features/ (Total: ~500 MB)
   ├── birdclef/{train,val,test}/*.npz
   ├── nips4bplus/validation/*.npz
   ├── species_list_train.json
   ├── species_list_validation_overlap.json
   └── label_map.json
```

### 1. Configuration & Taxonomy Mapping

#### [MODIFY] [species_config.py](file:///e:/Ornitho-Ex/src/config/species_config.py)
- Add BirdCLEF 2021 dataset path constants:
  - `BIRDCLEF_DIR = "/kaggle/input/birdclef-2021"`
  - `BIRDCLEF_METADATA = "/kaggle/input/birdclef-2021/train_metadata.csv"`
  - `BIRDCLEF_AUDIO_DIR = "/kaggle/input/birdclef-2021/train_short_audio"`
- Update feature extraction constants:
  - `sample_rate`: 32000 (native rate of BirdCLEF audio, or 44100 resampled)
  - `n_mels`: 128
  - `hop_length`: 512
  - `n_fft`: 2048
  - `window_duration_sec`: 4.0
- Add helper dictionary mapping NIPS4Bplus scientific names to BirdCLEF `primary_label` codes.

---

### 2. Preprocessing Notebook Overhaul

#### [MODIFY] [phase1_bird_xai_preprocessing.py](file:///e:/Ornitho-Ex/notebooks/phase1_bird_xai_preprocessing.py)

- **Cell 1 & 2 (Setup & Paths):**
  - Point input paths to `/kaggle/input/birdclef-2021/`.
  - Output paths stay in `/kaggle/working/processed_features/`.
  - Remove external network dependencies and API key checks.

- **Cell 3 & 4 (Replace API Query with Metadata Filter):**
  - Remove Xeno-canto HTTP request functions (`xc_api_query`, `fetch_all_recordings`).
  - Read `/kaggle/input/birdclef-2021/train_metadata.csv`.
  - Filter by `scientific_name` and `rating >= 3.5`.
  - Balance dataset: sample up to `MAX_CLIPS = 50` per species.

- **Cell 5 (Dataset Manifest Preparation):**
  - Generate the training manifest directly from filtered BirdCLEF rows.
  - Print species counts, accepted classes, and distribution summary.

- **Cell 6 (Fast Local Feature Extraction — replaces raw download):**
  - Iterate through local `.ogg` files in `/kaggle/input/birdclef-2021/train_short_audio/`.
  - For each audio file:
    1. Load audio with `soundfile` / `librosa`.
    2. Segment into 4.0s active windows.
    3. Compute log-mel spectrogram (`128 × 345`, `np.float16`).
    4. Compute 6 programmatic acoustic concepts:
       - `peak_frequency`
       - `trill_rate`
       - `call_duration`
       - `fm_rate`
       - `spectral_centroid`
       - `inter_call_silence`
    5. Save directly as compressed `.npz` files in `/kaggle/working/processed_features/birdclef/`.
  - Zero raw audio files written to disk. Disk footprint: **< 500 MB**.

- **Cell 7 & 8 (NIPS4Bplus Audio Alignment):**
  - Apply the identical feature extraction (log-mel + 6 concepts) to NIPS4Bplus validation cut segments.
  - Save all timestamp metadata in `nips4bplus_features_manifest.csv`.

- **Cell 9 & 10 (Metadata, Overlap & Disk Audit):**
  - Compute exact species overlap between BirdCLEF training classes and NIPS4Bplus validation classes.
  - Save `species_list_train.json`, `species_list_validation_overlap.json`, `label_map.json`.
  - Assert total disk consumption is `< 2.0 GB` with `shutil.disk_usage()`.

---

### 3. Documentation & State Tracking

#### [MODIFY] [trace.md](file:///e:/Ornitho-Ex/trace.md)
- Log decision to pivot training source to BirdCLEF 2021 to resolve the Kaggle 20 GB disk limit and remove live API download bottlenecks.

#### [MODIFY] [memory.md](file:///e:/Ornitho-Ex/memory.md)
- Update current status, input dataset requirements, and architecture notes.

---

## Verification Plan

### Automated Verification
1. **Compilation Check:**
   ```powershell
   python -m py_compile src/config/species_config.py
   python -m py_compile notebooks/phase1_bird_xai_preprocessing.py
   ```
2. **Offline Logic Validation:**
   - Verify that all imports (`soundfile`, `librosa`, `scipy`, `numpy`, `pandas`) resolve cleanly.
   - Verify programmatic concept extraction function outputs non-NaN, finite values on dummy audio signals.

### Kaggle Execution Verification
1. Attach `birdclef-2021` to the Kaggle notebook.
2. Run notebook cells 1–5: confirm metadata loads and species filter completes in < 5 seconds.
3. Run cell 6: observe local feature extraction. Should process ~2,000 clips in ~10–15 minutes with total `/kaggle/working/` disk usage staying under 1 GB.
4. Verify `bird-xai-processed-features` output dataset is created cleanly with status **Successful**.
