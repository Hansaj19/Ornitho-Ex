**Ques. Clarify this doubt, what are the proprocessing steps we did, and why we did them, also mention how they are given input to the models and what model we ar using and why?**

- Below is the clear, structured breakdown answering each of your questions.

---

### 1. What Preprocessing Steps Did We Do and WHY?

| # | Preprocessing Step | What We Did | WHY We Did It (The Scientific Reason) |
| --- | --- | --- | --- |
| **1** | **Quality Filtering & Class Capping** | Kept clips with `rating >= 3.5` and capped each species at `MAX_CLIPS = 50`. | **Eliminates microphone noise:** Crowd-sourced audio contains wind buffeting and loud human speech. Filtering ensures the model learns bird calls, not microphone hiss. Capping at 50 prevents majority species (like House Sparrows) from dominating rare species. |
| **2** | **Audio Resampling (32,000 Hz) & Mono Conversion** | Converted stereo to mono $(L+R)/2$ and resampled to 32 kHz. | **Captures full bird acoustic range:** By the Nyquist theorem, 32 kHz captures up to 16 kHz, covering all European bird calls (400 Hz to ~10,000 Hz) without wasting RAM on ultrasound. Mono eliminates microphone directional bias. |
| **3** | **Fixed 4.0-Second Segmentation** | Sliced long audio files into non-overlapping 4.0s windows ($128,000$ audio samples). | **Batched matrix processing:** Bird vocalizations (strophes and motifs) typically last 0.5–3.5s. Fixed 4s windows capture complete phrases while allowing GPUs to run parallel matrix multiplication without variable-length padding overhead. |
| **4** | **Log-Mel Spectrogram Extraction** | Converted 1D sound waves into 2D frequency-time images ($128$ mel bins $\times$ $250$ time steps) stored as `float16`. | **Matches biological hearing & vision backbones:** 1D waveforms have huge phase sensitivity. Mel spectrograms compress dynamic range logarithmically (matching how ears perceive pitch/loudness) and turn audio into a 2D image suitable for standard computer vision backbones. `float16` cuts RAM usage by 50%. |
| **5** | **Programmatic Acoustic Concept Extraction** | Computed 6 deterministic physical properties per 4s window (`peak_frequency`, `trill_rate`, `call_duration`, `fm_rate`, `spectral_centroid`, `inter_call_silence`). | **Bypasses the $50,000 annotation barrier:** Normal Concept Bottleneck Models require human ornithologists to manually label concepts for every clip. We used digital signal processing (FFT peaks, Hilbert envelopes) to extract ground-truth physics automatically with **zero human labeling cost**. |
| **6** | **Concept Standardization (Train Split Only)** | Standardized each concept: $c_{\text{norm}} = (c - \mu_{\text{train}}) / \sigma_{\text{train}}$. | **Prevents loss dominance:** Frequency is ~2,300 Hz while duration is ~0.67s. Without standardization, frequency loss would be $10,000\times$ larger and drown out all other concepts. Fitting solely on the train split prevents test data leakage. |
| **7** | **NIPS4Bplus Benchmark Extraction** | Sliced 1,687 validation recordings into 4s windows while preserving exact call start/end timestamps. | **Ground-truth explainability benchmark:** Standard datasets only tell you *which* bird is singing. NIPS4Bplus tells you *when* the bird is singing, allowing us to mathematically test if Grad-CAM heatmaps highlight actual bird calls or background silence. |

---

### 2. How Are They Given as Input to the Models?

```
Raw Audio (4s @ 32 kHz) ──▶ Mel Spectrogram ──▶ Tensor Shape: (Batch_Size, 1, 128, 250)
                                                                 │          │   │    └── 250 Time Frames
                                                                 │          │   └────── 128 Mel Bins
                                                                 │          └────────── 1 Mono Channel
                                                                 └───────────────── 128 Clips/Batch
```

- **In-Memory RAM Preloading:** In Notebook 2, all 74,684 spectrograms and concept vectors were preloaded into Kaggle RAM (~4.8 GB). This eliminated 59,000+ disk reads per epoch and accelerated training by ~15x.
- **DataLoader Output:** Each mini-batch yields:
  - `x`: `torch.FloatTensor` of shape `(128, 1, 128, 250)`
  - `c`: `torch.FloatTensor` of shape `(128, 6)` (normalized ground-truth concepts)
  - `y`: `torch.LongTensor` of shape `(128,)` (species label index `[0 .. 181]`)
- **Crucial Difference: Training Mode vs. Inference Mode:**
  - **During Training:** The model receives `x` and evaluates its predictions against both `y` and `c` using the multi-task joint loss.
  - **During Inference / Evaluation:** **The model NEVER receives `c`!** It receives **only the audio spectrogram `x`**. The model predicts its own concept vector $\hat{c}$, and the species classifier decides the species **strictly from $\hat{c}$**.

---

### 3. What Models Are We Using and WHY?

Both models share the exact same convolutional backbone to ensure fair, unconfounded comparisons:

#### Backbone: EfficientNet-B0 (`timm`, pretrained on ImageNet)

- **Why?** It achieves state-of-the-art representation capacity with only **~4.0 million parameters** (compared to ResNet-50's 25M). It fits easily in Kaggle GPU VRAM and uses standard 2D convolution layers that directly support gradient-based explainability tools (Grad-CAM, Captum). We set `in_chans=1` to adapt the first convolution layer from RGB images to 1-channel spectrograms.

#### Model 1: Black-Box Baseline

- **Architecture:** `EfficientNet-B0 (1280 features) ──▶ Dropout(0.3) ──▶ Linear(1280 ──▶ 182 species)`
- **Why?** Serves as our **empirical performance ceiling (56.26% accuracy)**. It represents standard opaque deep learning classifiers and acts as the subject for our post-hoc explainability tests (Grad-CAM and Integrated Gradients).

#### Model 2: Concept Bottleneck Model (CBM)

- **Architecture:**

  ```
  Input Spectrogram (1, 128, 250)
        │
  EfficientNet-B0 Encoder (1280 features)
        │
  Concept Head: Linear(1280 ──▶ 256) ──▶ BatchNorm ──▶ ReLU ──▶ Dropout ──▶ Linear(256 ──▶ 6 concepts)
        │
  Predicted Concepts ĉ in ℝ⁶ (Pitch, Trill, Duration, FM Rate, Centroid, Silence)
        │
  Species Head: Dropout(0.2) ──▶ Linear(6 ──▶ 182 species)
  ```

- **Why? The Strict Bottleneck Principle:**
  - The species classifier is **forced** to make predictions using **ONLY** the 6 acoustic concepts. There are **no skip connections** and no backdoor channels to the 1,280-dim latent embedding.
  - **Causally Load-Bearing Explanations:** Explanations are no longer post-hoc guesses. Every species prediction is an explicit linear combination of the 6 concepts:
    $$\text{logit}(\text{Species}_i) = w_{i,1} \cdot \text{pitch} + w_{i,2} \cdot \text{trill} + \dots + b_i$$
  - **Test-Time Intervenability:** An ornithologist can manually adjust a concept (e.g., dial frequency from 2 kHz to 5 kHz) and watch the model update its prediction live.

#### The Joint Loss & $\lambda$ (Lambda) Sweep

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(\text{CrossEntropy}) + \lambda \cdot \mathcal{L}_{\text{concept}}(\text{Smooth L1 / Huber})$$

- By sweeping $\lambda \in \{0.0, 0.1, 0.5, 1.0, 5.0\}$, we established that **$\lambda = 0.5$ is the Pareto sweet spot**: it reduces concept error by **82%** (down to MAE 1.70) while retaining **95%** of the black-box baseline accuracy.

---

### 4. Notebook 3 Has Been Created

The evaluation notebook has been created directly as an `.ipynb` file:
📁 **[`notebooks/bird_xai_evaluation.ipynb`](file:///e:/Ornitho-Ex/notebooks/bird_xai_evaluation.ipynb)**

#### What Notebook 3 Performs

1. **Dependency & Environment Setup:** Installs `captum`, `grad-cam`, `timm`, `pytorch-lightning`, and `seaborn`.
2. **Robust Multi-Dataset Discovery:** Automatically discovers both attached Kaggle datasets (`ex-ai_bird_preprocessing` and `bird-xai-checkpoints`).
3. **Weight Restoration:** Loads the baseline checkpoint and all 5 CBM models ($\lambda \in \{0.0, 0.1, 0.5, 1.0, 5.0\}$).
4. **Quantitative Held-Out Testing:** Computes test accuracy, macro-F1, and concept MAE in both normalized units and real physical units (Hz, seconds).
5. **Post-Hoc Saliency Visualizations:** Computes **Grad-CAM** and **Integrated Gradients** via Captum and overlays them onto log-mel spectrograms.
6. **Faithfulness Benchmarking on NIPS4Bplus:**
   - **Deletion AUC:** Measures accuracy decay as top-attribution pixels are removed.
   - **Insertion AUC:** Measures accuracy recovery as top-attribution pixels are restored.
   - **Temporal Call Alignment:** Calculates what percentage of heatmap energy lands strictly inside the bird's vocalization window vs. background noise.
7. **CBM Causal Interventions:** Executes a real-time intervention sweep on `peak_frequency` to demonstrate that concept changes causally alter species predictions.
8. **Publication-Ready Figures:** Exports high-resolution figures to `/kaggle/working/figures/` (Pareto curve, perturbation curves, concept weight matrices) and saves all metrics to `/kaggle/working/evaluation_metrics.csv`.
