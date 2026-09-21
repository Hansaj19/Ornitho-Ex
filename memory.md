# 🧠 Ornitho-Ex: Agent Memory & Project Preferences

> **Purpose:** Quick-handover document for any new LLM agent taking over this project. Contains owner preferences, project context, standing rules, and current status. **Must be kept up-to-date at every session.**
> **Last Updated:** 2026-09-17 (v4 — BirdCLEF 2021 pivot implemented)

---

## 🔁 Mandatory Agent Behaviors (Read First!)

1. **Always update `trace.md`** after any significant decision, architecture change, or problem resolution. Prepend new entries (newest at top).
2. **Always update `memory.md`** at the end of every session: update `Current Status`, `What's Done`, `What's Next`, and `Last Updated`.
3. **Always update `implementation_plan.md`** checkboxes as tasks are completed (`[ ]` → `[x]`).
4. **Never discard NIPS4Bplus timestamp metadata** — it is critical for faithfulness evaluation.
5. **Never mix training and validation species lists** — strictly separate code paths.
6. **Log all experiments to W&B** with full config (λ, seed, model type).
7. **Read `implementation_plan.md` first** at the start of any session to understand current progress.

---

## 👤 Owner Preferences

| Preference | Value |
|---|---|
| Language | Python 3.10+ |
| Deep Learning Framework | PyTorch + PyTorch Lightning |
| Experiment Tracking | Weights & Biases (`wandb`) |
| Code Style | PEP8, type hints preferred |
| Config Management | Central `config.yaml` (not hardcoded values) |
| Verbosity | Detailed comments in code; explicit variable names |
| Testing | Sanity checks and visualizations before full training runs |
| Documentation | Keep `trace.md`, `memory.md`, `implementation_plan.md` updated regularly |

---

## 📁 Project Root

```
e:\Ornitho-Ex\
```

---

## Execution Environment: Kaggle Notebooks

> [!IMPORTANT]
> **This project does NOT run locally.** Local hardware has no CUDA. Training a 15-25 run lambda-sweep requires GPU. Use Kaggle free-tier (30 hrs/week GPU quota).

| Notebook | Name | GPU | Purpose |
|---|---|---|---|
| 1 | `bird-xai-preprocessing` | OFF | BirdCLEF 2021 local feature extraction + NIPS4Bplus cutting |
| 2 | `bird-xai-training` | ON | Train CBM + black-box baseline, full lambda sweep |
| 3 | `bird-xai-evaluation` | ON (light) | Grad-CAM, faithfulness metrics, results plots |

**Kaggle Datasets (data hand-off between notebooks):**
- `bird-xai-processed-features` — output of Notebook 1, input of Notebooks 2 & 3
- `bird-xai-checkpoints` — output of Notebook 2, input of Notebook 3

**Path conventions:**
- Read-only: `/kaggle/input/<dataset-name>/`
- Writable: `/kaggle/working/`
- No local paths (`C:\...`, `/home/...`) allowed in generated code

---

## 📐 Project Overview (Brief)

Build a **Concept Bottleneck Model (CBM)** for bird species classification from audio (log-mel spectrograms). The model first predicts human-interpretable acoustic **concepts** (trill rate, peak frequency, call duration, FM rate, spectral centroid, inter-call silence) before making species predictions — providing inherently interpretable, causally faithful explanations.

**Core novelty:** Concept targets are **programmatically extracted** (not manually labeled). The **λ sweep** (joint loss weight) quantifies the accuracy-vs-interpretability tradeoff.

**Datasets:**
- **BirdCLEF 2021** (Kaggle) → Training (European focus, ~70 species, rating ≥ 3.5)
- **NIPS4Bplus** → Faithfulness validation only (51 European species, timestamped)

---

## 🏗️ Architecture Summary

```
Audio → Log-mel spectrogram → EfficientNet-B0 → ConceptBottleneck head → Species classifier
                                              ↘ (Baseline) → Direct softmax → Grad-CAM/IG
```

**Loss:** `L = L_task + λ * L_concept`  
**λ sweep:** {0, 0.1, 0.5, 1.0, 5.0} × {3–5 seeds}

---

## 🧪 Key Experiments

| Experiment | Description |
|---|---|
| λ sweep | Core contribution: accuracy-vs-interpretability curve |
| Joint vs. sequential training | Ablation: freeze encoder before training classifier |
| Test-time concept intervention | Validate causal meaningfulness of concepts |
| BirdNET encoder swap | Optional: test bottleneck gains with bird-domain pretraining |

---

## ✅ Current Status

**Phase:** 1 — Preprocessing (BirdCLEF 2021 pivot complete, ready for Kaggle execution)  
**What's Done:**
- Project brief analyzed (v1, v2, and Kaggle workflow guidelines doc)
- Implementation plan created and updated (v3 → v4)
- `trace.md`, `memory.md` kept up-to-date
- **Pivoted Notebook 1** from Xeno-canto REST API to BirdCLEF 2021 Kaggle dataset
  - `src/config/species_config.py` — BirdCLEF path constants, taxonomy map, feature params
  - `notebooks/phase1_bird_xai_preprocessing.py` — full overhaul; no network calls
  - Segmentation: 4.0 s windows at 32 kHz → (128, 250) log-mel + 6 concept targets
  - Output: compressed `.npz` in `/kaggle/working/processed_features/` (~500 MB expected)

**What's Next:**
- **Kaggle:** Attach `birdclef-2021` dataset and run Notebook 1 (cells 1–11)
- Verify disk usage < 2 GB and species overlap > 30 species
- Save Notebook 1 output as `bird-xai-processed-features` dataset
- Phase 2: Build DataModule and EfficientNet-B0 backbone (Notebook 2)
- **Phase 4A (early gate):** Minimal CNN smoke-test must pass before full CBM work

**Blockers / Open Questions:**
- BirdCLEF 2021 species coverage not yet verified (exact overlap with NIPS4Bplus TBD at runtime)
- Window size fixed at 4 s (decision logged in trace.md)
- 6 concepts defined — can be expanded after smoke-test
- W&B API key needs to be set as Kaggle Secret before running Notebook 2

**Input Datasets Required (Kaggle sidebar):**
- `birdclef-2021` → competition dataset (attach via '+ Add Data')
- `nips4b-audio-raw` → NIPS4B WAV files
- `nips4bplus-annotations` → NIPS4Bplus CSV annotations

---

## 🗂️ Key Files

| File | Purpose |
|---|---|
| [`Bird_Species_Detection_XAI_Project.md`](file:///e:/Ornitho-Ex/Bird_Species_Detection_XAI_Project.md) | Original project brief — source of truth |
| [`implementation_plan.md`](file:///C:/Users/patid/.gemini/antigravity-ide/brain/867914d6-9a20-4fa3-bd8a-0f049c312ca2/implementation_plan.md) | Phased plan with checkboxes — track progress here |
| [`trace.md`](file:///e:/Ornitho-Ex/trace.md) | Decision and change log — update after major decisions |
| [`memory.md`](file:///e:/Ornitho-Ex/memory.md) | This file — agent handover context |
| `config.yaml` | Central config (to be created in Phase 0) |
| `docs/species_list_train.csv` | Training species list (to be created in Phase 1) |
| `docs/species_list_val_nips4b.csv` | Validation species overlap list (to be created in Phase 1) |

---

## ⚠️ Critical Non-Negotiables (from Project Brief §8 — v2)

1. Never mix training and validation species scope.
2. Concept targets are programmatic — state this explicitly in the report.
3. Run every λ setting across 3–5 seeds.
4. Use paired significance test (paired t-test or Wilcoxon) for accuracy comparison.
5. Black-box baseline must use **identical** EfficientNet-B0 backbone.
6. Never discard NIPS4Bplus timestamp metadata.
7. Report λ tradeoff as a **curve**, not a single number.
8. Validate concept causality via **test-time intervention**.
9. **Start simple:** Build minimal CNN smoke-test before full CBM — pipeline must be proven before adding architecture complexity.
10. **Monitor train-vs-val every epoch:** Dropout/pooling first, architectural changes last.
11. **Stratified split** for all Xeno-canto train/val/test divisions.
12. **Freefield1010 is explicitly deferred** — do not implement.

---

## 🔄 Session Log

| Date | Agent | Session Summary |
|---|---|---|
| 2026-08-17 | Antigravity (Claude Sonnet 4.6 Thinking) | Project initialized. Created implementation plan, trace.md, memory.md. No code written yet. |
| 2026-08-17 | Antigravity (Claude Sonnet 4.6 Thinking) | Owner made 6 edits to project brief (v2). All 3 tracking docs updated. |
| 2026-08-18 | Antigravity (Claude Sonnet 4.6 Thinking) | Read Kaggle_Workflow_and_Coding_Guidelines.md. Full Kaggle environment integrated into all 3 tracking docs: 3-notebook structure, /kaggle/ paths, checkpointing, GPU quota rules, real-time results_summary.csv logging. Plan updated to v3. |
| 2026-09-17 | Antigravity (Claude Sonnet 4.6 Thinking) | Implemented BirdCLEF 2021 pivot plan. Overhauled species_config.py (BirdCLEF paths, taxonomy map, new feature constants). Fully rewrote phase1_bird_xai_preprocessing.py (11 cells): removed all Xeno-canto API code, added local metadata filter + stratified split + on-the-fly log-mel+concept extraction → .npz. Updated trace.md and memory.md. |

---

*New agents: append your session to the Session Log above, and update Current Status before ending.*
