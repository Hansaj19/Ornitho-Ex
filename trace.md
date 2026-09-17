# 📋 Ornitho-Ex: Decision & Change Trace Log

> **Purpose:** Records all important decisions, architectural choices, and changes made during the project, along with their rationale. Updated at every significant milestone or decision point.
> **Format:** Newest entries at the top.

---

## How to Use This File
- Add an entry whenever a significant decision is made, a design is changed, or a problem is resolved.
- Each entry: `## [Date] — [Short Title]` followed by **Decision**, **Reason**, and **Impact**.

---

## [2026-08-18] — Kaggle Disk Space Exhaustion Fix

**Context:** The overnight Kaggle run failed with `OSError: [Errno 28] No space left on device` during notebook export. Kaggle has a strict 20GB output limit in `/kaggle/working/`. 
**Issue:** 71 species * 150 clips = 10,650 audio clips. Since some Xeno-canto files are multi-minute recordings (e.g. 15 mins = ~30MB), the total audio size exceeded 20GB.
**Decision:**
1. Lowered `MAX_CLIPS` from 150 to 100 per species.
2. Implemented a `MAX_LENGTH_SEC` filter (120 seconds) in `fetch_all_recordings` to ignore excessively long recordings.
**Impact:** Total clips lowered to ~7,100, and average file size reduced by dropping huge recordings. This guarantees the Phase 1 dataset stays well below the 20GB Kaggle limit. Phase 1 notebook and `species_config.py` updated.

---

## [2026-08-18] — Xeno-canto API v3 Upgrade

**Decision:** Switch from Xeno-canto API v2 to the new API v3 (`https://xeno-canto.org/api/3/recordings`).
**Reason:** Owner noted the API v3 changes (requires an API key, strictly accepts search tags, removes rate limits, lowers default page size to 100).
**Impact:**
- `xc_api_query` in Phase 1 notebook updated to dynamically parse species names into exact `gen:` and `sp:` search tags (e.g., `gen:"Turdus" sp:"merula"`).
- API request now requires passing `key`.
- Notebook uses `Kaggle Secrets` to securely load `XC_API_KEY` (with a fallback string if missing). Owner must generate a key on xeno-canto.org and add it to Kaggle Secrets before running the notebook.

---

## [2026-08-18] — NIPS4Bplus: Confirmed Sources & Cut Script Implementation

**Context:** Owner provided the official NIPS4Bplus usage instructions from the paper/GitHub repo.

**Confirmed source URLs:**
- NIPS4B audio WAVs: `http://sabiod.univ-tln.fr/nips4b/media/birds/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV.tar.gz` (~1.6 GB)
- NIPS4B train labels (species list CSV): `http://sabiod.univ-tln.fr/nips4b/media/birds/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS.tar`
- NIPS4Bplus annotation CSVs (figshare): `https://figshare.com/ndownloader/articles/6798548/versions/3` (zip of per-recording CSVs)
- Cutting script: `https://raw.githubusercontent.com/fbravosanchez/NIPS4Bplus/master/cut_nips4bplus_files.py`
- List generation script: `https://raw.githubusercontent.com/fbravosanchez/NIPS4Bplus/master/generate_file_lists.py`
- GitHub repo: `https://github.com/fbravosanchez/NIPS4Bplus`

**Decision:** Implement `cut_nips4bplus_files.py` logic **inline** in the notebook rather than calling it as a subprocess.
**Reason:** More robust in Kaggle (no PATH issues), allows full control over column name normalization (CSV column names vary slightly), and makes timestamp preservation explicit in code with a comment rather than implicit.

**Decision:** Download all NIPS4B files directly in Kaggle (internet ON) — no manual Kaggle Dataset upload needed.
**Reason:** All files are at stable public URLs that can be downloaded via `requests` with retry/backoff. Only the ~1.6 GB WAV tar is large; annotated as idempotent so it can be re-run if it times out.

**Annotation CSV format (from script inspection):**
- One CSV per NIPS4B training recording
- Columns include: Starttime, Stoptime, species/label (exact names may vary — code normalizes)
- `nips4b_birdchallenge_espece_list.csv` (in labels tar) is the master species list

**Impact:** Cell 7 now downloads all 4 sources automatically. Cell 8 implements cutting inline. `nips4bplus_cut_manifest.csv` output carries all timestamp metadata (start_sec, end_sec per segment) — critical for faithfulness evaluation. Phase 1 notebook updated to v2.

---

## [2026-08-18] — New Document: Kaggle Workflow & Coding Guidelines

**Decision:** The entire project runs on **Kaggle Notebooks** (not locally). Local hardware has no CUDA support; CPU fallback is too slow for a 15–25 run λ-sweep.

**Key decisions from this document:**

### Notebook split (3 separate notebooks)
- `bird-xai-preprocessing` — CPU only, no GPU quota; queries Xeno-canto API (Internet ON required)
- `bird-xai-training` — GPU on; reads from `bird-xai-processed-features` Kaggle Dataset
- `bird-xai-evaluation` — GPU on (lighter); reads from both processed features + checkpoints
**Reason:** Don’t combine preprocessing and training into one notebook — wastes GPU quota on CPU-bound work and risks losing processed data if a GPU session crashes.

### Checkpointing strategy
- `save_top_k=1` (best model for evaluation) + `save_last=True` (latest for resuming after crash)
- Resume logic: check for `last.ckpt` before starting any fresh run
- **Reason:** Kaggle sessions can time out mid-sweep; with 15–25 runs, checkpointing is non-optional.

### Results logging
- Log every completed run to `results_summary.csv` **immediately** after it finishes (not batched at notebook end)
- **Reason:** Partial progress must survive session interruptions.

### GPU quota management
- Preprocessing notebook: GPU **OFF** — does not touch the 30hr/week free tier
- Training notebook: Run 1 epoch on 1 config first, extrapolate total time, confirm it fits quota
- Two GPU quota management bullet points were removed by owner — quota guidance simplified to the above two rules only.

### Kaggle coding conventions (10 rules for all code)
1. `/kaggle/input/<dataset>/` is read-only; `/kaggle/working/` is writable
2. No hardcoded local paths
3. Always check GPU availability explicitly
4. Xeno-canto API: retry + exponential backoff
5. No persistent state between sessions
6. Set all seeds (torch, numpy, random, cuda)
7. Preprocessing: GPU accelerator OFF
8. Preprocessing notebook header: `# IMPORTANT: Enable Internet (Settings > Internet > ON)`
9. Log every run to `results_summary.csv` immediately

**Impact:** All code generation must follow Kaggle path conventions. Local `data/` directory removed from project structure (Kaggle datasets replace it). Implementation plan updated to v3.

---

## [2026-08-17] — Owner Edits to Project Brief (v2 — 6 Changes)

**Changes made by owner directly to `Bird_Species_Detection_XAI_Project.md`:**

### 1. Freefield1010 explicitly deferred
**Decision:** Freefield1010 is **not needed to implement now** (owner changed label from "optional" to "Not needed to implement now").
**Reason:** The bird-presence pre-filtering stage is not part of the core novelty. Adds scope without contributing to the XAI comparison.
**Impact:** Phase 1C removed from active checklist. Freefield1010 will only be revisited if the pipeline produces excessive false positives on background noise.

### 2. Multi-channel spectral stacking added as optional ablation (§3)
**Decision:** Spectral bandwidth, spectral centroid, chroma STFT, and raw STFT can be stacked as separate input channels alongside log-mel (analogous to RGB).
**Reason:** Gives the CNN richer input than mel spectrogram alone, but adds preprocessing complexity and another failure point to debug.
**Impact:** Treat as **ablation only** — default stays single-channel log-mel. Added as Phase 2D in plan with a clear "do this last" warning.

### 3. Padding & normalization rules added (§3)
**Decision:** Zero-pad features to a common target shape (no truncation). Normalize in two passes: min-max to [0,1] then divide by std. Fit scaler only on train split.
**Reason:** Fixed-length windowing won't guarantee uniform shapes when stacking multi-resolution feature types. Two-pass normalization + train-only fitting avoids data leakage.
**Impact:** Added as Phase 2B-bis in plan. A dedicated `src/features/normalize.py` must be implemented.

### 4. Start simple — minimal CNN smoke-test first (Guideline 9)
**Decision:** Before building the full CBM, implement a minimal CNN (few conv layers, no bottleneck) to confirm the pipeline works end-to-end.
**Reason:** Jumping straight to the full CBM makes it hard to isolate whether a bug is in the data pipeline or the model architecture.
**Impact:** Added as Phase 4A (Smoke-Test) — it must pass before any CBM work begins.

### 5. Per-epoch train-vs-val monitoring + overfitting defense (Guideline 10)
**Decision:** Monitor train and validation curves every epoch. Use Dropout and pooling as the first-line overfitting defense before trying architectural changes.
**Reason:** Judging a model only on its best epoch masks overfitting. Dropout is cheap; architectural changes are expensive and hard to debug.
**Impact:** Training scripts must log both train and val metrics per epoch. Dropout layers added as default in CBM and black-box models.

### 6. Stratified split mandatory (Guideline 11)
**Decision:** Use stratified train/val/test split for Xeno-canto data (stratify by species label).
**Reason:** Species counts are naturally imbalanced. Random splits risk val/test sets missing rare species entirely.
**Impact:** `sklearn.model_selection.train_test_split(..., stratify=labels)` must be used in the DataModule.

---

## [2026-08-17] — Project Initialization

**Decision:** Created `implementation_plan.md`, `trace.md`, and `memory.md` as project scaffolding documents.

**Reason:** To ensure clear planning, traceability, and quick agent handover capability from the start, as required by project owner.

**Impact:** All future work will follow the phased plan in `implementation_plan.md`. Every architectural or parameter decision will be logged here.

---

## [2026-08-17] — Architecture Decision: EfficientNet-B0 as Backbone

**Decision:** Use EfficientNet-B0 (ImageNet-pretrained via `timm`) as the shared encoder for both the CBM and the black-box baseline.

**Reason (from project brief):**
- Best accuracy-to-compute ratio for repeated experiments (λ-sweep × multiple seeds).
- Most mature Grad-CAM / Captum tooling support.
- Clean insertion point for the concept bottleneck at the final feature map.
- **Not** Wav2Vec2/HuBERT: those produce waveform-level attributions, less interpretable as spectrogram heatmaps.

**Impact:** Both models share identical backbone architecture — ensuring accuracy comparison is not confounded by encoder differences.

---

## [2026-08-17] — Architecture Decision: Programmatic Concept Targets

**Decision:** Concept bottleneck targets (peak frequency, trill rate, call duration, FM rate, spectral centroid, inter-call silence) are extracted programmatically from audio signals — **no manual annotation**.

**Reason:** Addresses the common CBM criticism of expensive human annotation cost. All concept targets are computed algorithmically per segment using `librosa`-based feature extraction.

**Impact:** Must be explicitly stated in report to address annotation-cost criticism. Also means concept quality depends on the accuracy of programmatic extraction — validate carefully.

---

## [2026-08-17] — Data Strategy Decision: Two-List Species Strategy (Option B)

**Decision:** Decouple training species list (broad, 50–100+ species from Xeno-canto) from faithfulness validation species list (narrow, NIPS4Bplus-overlap subset only).

**Reason:** NIPS4Bplus only covers 51 European species. Restricting the full training scope to fit this dataset would artificially shrink the project. The two-list approach preserves realistic classification scope while still enabling ground-truth faithfulness evaluation on the subset.

**Impact:** Two separate CSV manifests must be maintained (`species_list_train.csv`, `species_list_val_nips4b.csv`). Code must never cross-contaminate these lists.

---

## [2026-08-17] — Experiment Design Decision: λ Sweep with Multiple Seeds

**Decision:** Run λ ∈ {0, 0.1, 0.5, 1.0, 5.0} × {3–5 random seeds}. Report as accuracy-vs-interpretability curve.

**Reason:** Bird audio data is noisy; single-run results are unreliable. λ sweep is the project's central empirical contribution. Reporting a single λ value understates the finding.

**Impact:** Automated sweep script needed. W&B logging must capture seed and λ per run. Statistical significance test (paired t-test or Wilcoxon) required on final comparison.

---

*[Future entries will be prepended here as decisions are made]*
