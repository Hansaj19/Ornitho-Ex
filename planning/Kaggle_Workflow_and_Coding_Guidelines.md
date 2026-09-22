# Kaggle Workflow, Notebook Structure & Coding Guidelines

## 1. System Requirements

| Component | Minimum (CPU-only, not recommended) | Recommended |
|---|---|---|
| RAM | 16GB | 16GB+ |
| GPU VRAM | N/A (CPU fallback) | 8GB+ dedicated (e.g. RTX 3060/4060, T4 or better) |
| Storage | 50–100GB free | Same, SSD preferred |
| CPU | Any modern multi-core | Same |

**Note:** Local hardware with <1GB VRAM or integrated/mobile GPUs (e.g. AMD Radeon integrated graphics) cannot realistically train this pipeline — no CUDA support, and CPU fallback is far too slow for a 15–25 run λ-sweep. This project is designed to run **entirely on Kaggle Notebooks**, using free GPU quota (30 hrs/week) only where needed.

---

## 2. Notebook Structure

Split the project into three separate Kaggle notebooks. Do not combine preprocessing and training into one notebook — this wastes GPU quota on CPU-bound work and risks losing processed data if a GPU session crashes.

### Notebook 1 — `bird-xai-preprocessing` (CPU only, no GPU quota used)

**Inputs:** none (pulls directly from the Xeno-canto API; requires "Internet" toggle ON in notebook settings)

**Does:**
- Queries Xeno-canto API for the selected ~50–100 species (training list)
- VAD, resampling to 44.1kHz, mono conversion, 3–5 sec segmentation
- Extracts log-mel spectrograms + concept-target features (peak frequency, trill rate, call duration, frequency modulation, spectral centroid, inter-call silence)
- Applies the same pipeline to NIPS4Bplus (cut via `cut_nips4bplus_files.py` using CSV timestamps), filtered to the species overlapping with the training list
- Preserves NIPS4Bplus timestamp metadata (required later for faithfulness evaluation — never discard)

**Outputs → publish as Kaggle Dataset `bird-xai-processed-features`:**
```
/kaggle/working/
├── xeno_canto/
│   ├── train/        (spectrograms + concept targets, .npy or .pt)
│   ├── val/
│   └── test/
├── nips4bplus/
│   └── validation/    (spectrograms + concept targets + timestamp metadata)
├── species_list_train.json
├── species_list_validation_overlap.json
└── metadata.csv       (file → species → split → source dataset)
```

### Notebook 2 — `bird-xai-training` (GPU on)

**Inputs:** attach `bird-xai-processed-features` dataset (read from `/kaggle/input/bird-xai-processed-features/`)

**Does:**
- Loads processed features
- Trains the black-box baseline (encoder → classifier) and the concept-bottleneck model (encoder → bottleneck → classifier)
- Runs the λ sweep (0, 0.1, 0.5, 1.0, 5.0) × 3–5 seeds
- Saves checkpoints during training (see Section 4)
- Logs every completed run to `results_summary.csv` immediately, not just at notebook end

**Outputs → publish as Kaggle Dataset `bird-xai-checkpoints`:**
```
/kaggle/working/
├── checkpoints/
│   ├── baseline_seed0.ckpt
│   ├── cbm_lambda0.1_seed0_last.ckpt
│   ├── cbm_lambda0.1_seed0_best.ckpt
│   └── ...            (one set per λ × seed combination)
├── training_logs/      (loss curves, W&B run IDs)
└── results_summary.csv (per-run: lambda, seed, accuracy, concept MAE, checkpoint path, status)
```

### Notebook 3 — `bird-xai-evaluation` (GPU on, lighter compute)

**Inputs:** attach both `bird-xai-processed-features` and `bird-xai-checkpoints`

**Does:**
- Loads trained checkpoints
- Runs Grad-CAM / Score-CAM / Integrated Gradients on the black-box baseline
- Computes deletion/insertion AUC and Grad-CAM alignment against NIPS4Bplus timestamped ground truth
- Runs the test-time concept intervention experiment
- Produces the final accuracy-vs-faithfulness results table and plots

---

## 3. Kaggle-Environment Coding Conventions (for Antigravity)

Paste this at the top of your Antigravity coding session so generated code matches the Kaggle environment:

```
Environment: Kaggle Notebooks. Follow these rules for all code generated in this project:

1. Use /kaggle/input/<dataset-name>/ for all read-only input data. Never attempt to write
   to this path — it is read-only and will raise an error.
2. Use /kaggle/working/ for all outputs. This is the only writable directory, and its
   contents become the notebook's output dataset when the notebook version is saved.
3. Do not hardcode local absolute paths (no /home/user/..., no C:\...). All paths must
   resolve relative to /kaggle/input/ and /kaggle/working/.
4. Always check for GPU availability explicitly:
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   Never assume a GPU is present — the preprocessing notebook runs CPU-only by design.
5. Wrap all Xeno-canto API calls with retry logic and exponential backoff — Kaggle's
   internet access can be intermittent under load.
6. Do not rely on persistent state between sessions. /kaggle/working/ resets on a fresh
   session unless the previous version's outputs were explicitly saved as a Dataset and
   re-attached as input.
7. Set random seeds explicitly (torch, numpy, random, and cuda) at the start of every
   training run, since the seed sweep depends on reproducibility.
8. Preprocessing notebook must not enable the GPU accelerator — it should run entirely
   on CPU to avoid consuming GPU quota unnecessarily.
9. State the "Internet must be ON" requirement as a comment at the top of the
   preprocessing notebook, since it is OFF by default in new Kaggle notebooks.
10. Log every completed training run (lambda, seed, metrics, checkpoint path) to
    results_summary.csv immediately after that run finishes — not batched at the end —
    so partial progress survives a session interruption.
```

---

## 4. Checkpointing Strategy (Training Notebook)

Kaggle sessions can be interrupted (timeouts, quota limits, disconnects) mid-run. Checkpointing is required, not optional, given the 15–25 run λ-sweep.

**PyTorch Lightning implementation:**

```python
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch, os

checkpoint_callback = ModelCheckpoint(
    dirpath=f"/kaggle/working/checkpoints/lambda{lambda_val}_seed{seed}/",
    filename="{epoch}-{val_loss:.3f}",
    save_top_k=1,          # keep only the best checkpoint per run
    monitor="val_loss",
    mode="min",
    save_last=True,        # always keep the most recent epoch, to resume after interruption
    every_n_epochs=1,
)

trainer = pl.Trainer(
    callbacks=[checkpoint_callback],
    max_epochs=50,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
)

# Resume logic — check for an existing checkpoint before starting a fresh run
ckpt_path = f"/kaggle/working/checkpoints/lambda{lambda_val}_seed{seed}/last.ckpt"
resume_from = ckpt_path if os.path.exists(ckpt_path) else None

trainer.fit(model, datamodule=dm, ckpt_path=resume_from)
```

**Why `save_top_k=1` + `save_last=True`:** Kaggle output storage per notebook version is limited. With 15–25 runs × multiple epochs, saving every epoch's checkpoint would bloat the output dataset unnecessarily. This keeps only the best model (for evaluation) and the last epoch (for resuming), which is sufficient for both purposes.

---

## 5. GPU Quota Management (30 hrs/week free tier)

- Preprocessing notebook: GPU accelerator **OFF** — does not touch the weekly quota, since VAD/resampling/spectrogram extraction are CPU-bound.
- Training notebook: GPU **ON**. Before committing to the full sweep, run a single epoch on one configuration and extrapolate total time needed, to confirm the full sweep fits inside the weekly quota.

