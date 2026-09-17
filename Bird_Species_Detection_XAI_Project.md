# Explainable Bird Species Detection from Audio

## 1. Objective and Abstract

**Objective:** Build a deep learning pipeline that classifies bird species from audio recordings, while making the model's decisions interpretable through a concept bottleneck architecture — instead of relying purely on post-hoc, black-box explanations.

**Abstract:**
Automated bird species identification from audio (bioacoustic monitoring) is widely used in ecological research and conservation, but current state-of-the-art models (CNN/Transformer-based) are black boxes — they offer no insight into *why* a species was predicted. This project proposes a Concept Bottleneck Model (CBM) for bird species classification, where the model first predicts human-interpretable acoustic concepts (trill rate, peak frequency, call duration, frequency modulation) before making its final species prediction. This forces explanations to be causally load-bearing rather than post-hoc rationalizations. The model is trained on a broad, multi-region Xeno-canto species set and evaluated for both classification accuracy and explanation faithfulness — the latter validated against NIPS4Bplus, a richly time-annotated European birdsong dataset. The project quantifies the accuracy-vs-interpretability tradeoff via a controlled loss-weighting sweep, providing an empirically grounded case for when and how much interpretability costs in real-world audio classification.

---

## 2. Datasets Used and How

| Dataset | Role | Why |
|---|---|---|
| **Xeno-canto** (via public API) | Primary training data | Global, large-scale, crowdsourced bird call recordings with species labels; used to train the full classification pipeline across a broad species set (Option B scope). |
| **NIPS4Bplus** | Faithfulness / explainability validation only | Provides 51 European bird species with **timestamped, frame-level call annotations** (687 recordings, ~1 hour audio, 44.1kHz) — the only dataset in this pipeline with ground-truth call timing, essential for scoring whether attribution maps point at real bird calls. |
| **Freefield1010** (Not needed to implement now) | Negative class / background noise | Used only if a "bird present vs. not" pre-filtering stage is added; not part of the core novelty. |

**Two-list species strategy (Option B):**
- **Training species list:** a broad set (50–100+ species) pulled from Xeno-canto, filtered to species with ≥80–100 high-quality (A/B rated) clips. This defines what the model can classify.
- **Validation species list:** the subset of the training list that overlaps with NIPS4Bplus's 51 European species. This subset alone is used for faithfulness metrics (deletion/insertion AUC, Grad-CAM alignment).

This decouples "what the model can classify" (broad, realistic) from "where we can prove the explanations are honest" (narrower, ground-truth-backed) — avoiding an artificial shrinkage of the project's scope to fit the one dataset with call-level timestamps.

---

## 3. Preprocessing of Datasets

**Xeno-canto (training data):**
1. Query API for selected species (scientific name), filter by quality rating (keep A/B only).
2. Resample all audio to 44.1kHz (matches NIPS4Bplus, keeps both datasets acoustically comparable).
3. Convert to mono.
4. Trim leading/trailing silence via energy-based VAD.
5. Segment each clip into fixed-length windows (3–5 sec) — since Xeno-canto only provides clip-level labels, this window becomes the unit of classification.
6. Extract log-mel spectrograms per segment (model input).

**NIPS4Bplus (validation-only data):**
1. Use the original NIPS4B wavefiles + NIPS4Bplus CSV annotations to cut precise, timestamped call segments (via the public `cut_nips4bplus_files.py` utility).
2. Retain timestamp metadata per segment — required for faithfulness scoring; never discarded.
3. Apply the same mono conversion + log-mel extraction as Xeno-canto, for feature comparability.
4. Filter to only the species overlapping with the training species list.

**Common feature extraction (both datasets):**
- Log-mel spectrogram (primary model input).
- Concept-target features computed programmatically per segment: peak frequency, trill repetition rate, call duration, frequency modulation rate, spectral centroid, inter-call silence duration. These serve as the "ground truth" for the concept bottleneck's auxiliary loss (no manual annotation needed).

**Optional enrichment — multi-channel spectral stacking:**
Rather than feeding the encoder a single-channel log-mel spectrogram, additional spectral features (spectral bandwidth, spectral centroid, chroma STFT, raw STFT) can be padded to a common shape and stacked as separate channels, analogous to RGB channels in an image. This gives the CNN richer input than mel spectrogram alone, at the cost of extra preprocessing complexity. Treat this as an ablation (single-channel log-mel vs. multi-channel stack) rather than a default, since it adds another preprocessing failure point to debug.

**Padding and normalization:**
- Since fixed-length windowing (3–5 sec) won't perfectly guarantee uniform array shapes across all extracted features (particularly if stacking multiple feature types with different native time/frequency resolutions), pad each feature to a common target shape via zero-padding rather than truncating, to avoid discarding call information.
- Normalize features in two passes: min-max scale to [0, 1] first, then standardize (divide by standard deviation) — do this per-split (fit on train, apply to val/test) to avoid data leakage.

---

## 4. Algorithms Used

| Component | Algorithm / Model | Purpose |
|---|---|---|
| Backbone encoder | **EfficientNet-B0** (ImageNet-pretrained, fine-tuned) | Extracts acoustic embeddings from log-mel spectrograms |
| Concept bottleneck | Custom linear/MLP head | Predicts interpretable acoustic concepts from encoder embeddings — this is the core novelty |
| Classifier head | Softmax classifier | Predicts species **only from the concept vector**, not directly from raw embeddings |
| Post-hoc XAI baseline | **Grad-CAM / Score-CAM** (via `pytorch-grad-cam`) | Explains the black-box (no-bottleneck) baseline model for comparison |
| Secondary attribution | **Integrated Gradients** (via Captum) | Additional post-hoc baseline, strengthens faithfulness comparison |
| Optional ablation encoder | BirdNET (pretrained embeddings) | Tests whether concept-bottleneck gains hold on top of a bird-domain-pretrained encoder |

**Why EfficientNet-B0:** best accuracy-to-compute ratio for repeated experiments (λ-sweep × multiple seeds), most mature Grad-CAM/Captum tooling support, and clean insertion point for the concept bottleneck at the final feature map.

**Why not Wav2Vec2/HuBERT as primary:** raw-waveform self-supervised models produce attributions on waveform samples, which are far less human-interpretable than spectrogram-region heatmaps — weaker fit for the explanation-readability goal of this project.

---

## 5. Integration of Algorithms

**Pipeline:**
`Audio → Preprocessing → Log-mel spectrogram → EfficientNet-B0 encoder → Concept bottleneck layer → Species classifier → (Species prediction, Concept scores)`

In parallel, a **black-box variant** (encoder → classifier directly, no bottleneck) is trained identically and explained post-hoc with Grad-CAM/Integrated Gradients, to serve as the baseline for the accuracy-vs-faithfulness comparison table.

**Joint training objective:**

```
L = L_task(g(c), y) + λ · L_concept(c, ĉ)
```

- `L_task`: cross-entropy loss between predicted species and true label
- `L_concept`: Huber/MSE loss between predicted concepts (c) and programmatically-extracted concept targets (ĉ)
- `λ`: tunable weight controlling the accuracy–interpretability tradeoff (swept across 0, 0.1, 0.5, 1.0, 5.0)

**Ablations to run:**
1. λ sweep (accuracy vs. concept fidelity vs. faithfulness, plotted as a curve).
2. Joint vs. sequential training (encoder+bottleneck frozen before classifier training).
3. Test-time concept intervention (manually edit a concept value, check if predicted species changes accordingly — validates causal meaningfulness of concepts).
4. Optional: BirdNET-embedding encoder swap, to test bottleneck gains independent of pretraining strength.

---

## 6. Dataflow Diagram and HLD Architecture

```
                    ┌──────────────────────────┐
                    │      Raw audio input     │
                    │  Xeno-canto / NIPS4Bplus │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │       Preprocessing      │
                    │  VAD, resample, denoise, │
                    │       segmentation       │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │     Feature extraction   │
                    │   Log-mel spectrogram    │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │      Acoustic encoder    │
                    │       EfficientNet-B0    │
                    └────────────┬─────────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 │                                │
     ┌───────────▼───────────┐       ┌────────────▼────────────┐
     │   Concept bottleneck  │       │   Direct classifier     │
     │  (trill rate, peak    │       │   (black-box baseline,  │
     │   freq, duration, ..) │       │    no bottleneck)       │
     └───────────┬───────────┘       └────────────┬────────────┘
                 │                                │
     ┌───────────▼───────────┐       ┌────────────▼────────────┐
     │   Species classifier  │       │  Grad-CAM / Integrated  │
     │  (from concepts only) │       │   Gradients (post-hoc)  │
     └───────────┬───────────┘       └────────────┬────────────┘
                 │                                │
     ┌───────────▼───────────┐       ┌────────────▼─────────────┐
     │  Species prediction + │       │   Attribution heatmap    │
     │  concept-based        │       │   (black-box explanation)│
     │  explanation          │       │                          │
     └───────────────────────┘       └──────────────────────────┘
                 │                                 │
                 └────────────────┬────────────────┘
                                  │
                    ┌─────────────▼─────────────────┐
                    │   Faithfulness evaluation     │
                    │  (NIPS4Bplus validation set,  │
                    │   deletion/insertion AUC,     │
                    │   concept intervention test)  │
                    └───────────────────────────────┘
```

**HLD summary:**
- **Data layer:** Xeno-canto (train) + NIPS4Bplus (validation-only, faithfulness).
- **Feature layer:** shared preprocessing + log-mel extraction pipeline for both datasets.
- **Model layer:** two parallel branches — interpretable (concept bottleneck) and black-box (baseline), sharing the same encoder architecture for fair comparison.
- **Explainability layer:** concept-based explanation (inherent) vs. Grad-CAM/IG (post-hoc), evaluated side by side.
- **Evaluation layer:** task metrics (accuracy, F1) + concept fidelity (MAE) + faithfulness (deletion/insertion AUC, intervention accuracy).

---

## 7. Tech Stack Used

| Layer | Tool / Library |
|---|---|
| Audio I/O & processing | `librosa`, `torchaudio` |
| Voice activity detection | `webrtcvad` / energy-based VAD |
| Deep learning framework | `PyTorch`, `PyTorch Lightning` |
| Backbone models | `timm` (EfficientNet-B0), BirdNET (optional) |
| Explainability | `Captum` (Integrated Gradients, TCAV), `pytorch-grad-cam` (Grad-CAM/Score-CAM) |
| Experiment tracking | `Weights & Biases` |
| Evaluation | `scikit-learn` (F1, confusion matrix), custom faithfulness metric scripts |
| Data querying | Xeno-canto REST API, NIPS4Bplus data-prep scripts (`cut_nips4bplus_files.py`) |

---

## 8. Important Guidelines for the Project

1. **Never mix training and validation species scope carelessly** — the training species list (broad, Xeno-canto) and the faithfulness validation species list (narrow, NIPS4Bplus-overlap subset) serve different purposes; keep them explicitly documented and separate in code and reporting.
2. **Concept targets are programmatically extracted, not manually labeled** — state this explicitly in the report, since it addresses a common criticism of concept bottleneck models (expensive annotation cost).
3. **Run every λ setting across 3–5 seeds** — bird audio data is noisy; single-run results will not hold up to scrutiny in evaluation or a patent filing.
4. **Use a paired significance test** (paired t-test or Wilcoxon) when comparing concept-bottleneck accuracy against the black-box baseline — a small accuracy drop must be shown to be statistically real, not noise.
5. **Keep the black-box baseline architecture identical to the encoder used in the bottleneck model** (same EfficientNet-B0 backbone) — otherwise the accuracy-vs-faithfulness comparison is confounded by architecture differences, not just the bottleneck itself.
6. **Do not discard NIPS4Bplus timestamp metadata during preprocessing** — it is the only source of ground-truth call timing and is required for deletion/insertion AUC and Grad-CAM alignment scoring.
7. **Report the accuracy-vs-interpretability tradeoff as a curve, not a single number** — the λ sweep is the project's central empirical contribution; a single λ value understates the finding.
8. **Validate concept causality via test-time intervention** — this is what distinguishes a genuinely interpretable model from one that merely predicts plausible-looking concepts without them mattering to the final decision.
9. **Start simple, add complexity only as needed** — begin with a minimal CNN (few conv layers, no bottleneck) to confirm the pipeline works end-to-end and can learn at all, then incrementally add depth/regularization. Jumping straight to the full concept-bottleneck architecture makes it hard to isolate whether a bug is in the data pipeline or the model.
10. **Monitor train-vs-validation curves every epoch, not just final metrics** — if training accuracy keeps climbing while validation accuracy plateaus or drops, that's overfitting; stop there rather than judging the model purely on its best epoch. Use Dropout and pooling layers as the first line of defense before reaching for architectural changes.
11. **Use a stratified split** when dividing Xeno-canto data into train/val/test, since species counts are naturally imbalanced — stratification keeps class proportions consistent across splits and avoids a val/test set that's accidentally missing rare species.