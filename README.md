# 🐦 Ornitho-Ex

**An Explainable Audio-Based Bird Species Recognition Framework for Biodiversity Assessment**

Ornitho-Ex uses a **Concept Bottleneck Model (CBM)** — EfficientNet-B0 backbone + 6 interpretable
acoustic concepts — trained on BirdCLEF+ 2026 to identify 182 European bird species from short
audio recordings.

---

## 🚀 Quick Start (local)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/ornitho-ex.git
cd ornitho-ex

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## 🐳 Docker

```bash
docker build -t ornitho-ex .
docker run -p 8501:8501 ornitho-ex
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (include models/checkpoints/cbm_lambda0.5_epoch09.zip).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point to pp.py as the entry point.
4. Streamlit Cloud will automatically install equirements.txt and packages.txt.

> **Note:** The checkpoint zip is ~50 MB. If it exceeds GitHub's 100 MB limit, use
> [Git LFS](https://git-lfs.com/) or store it on HuggingFace Hub and load it at runtime.

---

## 🤗 Deploy to HuggingFace Spaces

1. Create a new Space → **Streamlit** SDK.
2. Push all files including the checkpoint.
3. Add a README.md with the HF YAML front-matter if needed.

---

## 📁 Project Structure

```
ornitho-ex/
├── app.py                          ← Streamlit interface (main entry point)
├── requirements.txt                ← Python dependencies
├── packages.txt                    ← System packages (Streamlit Cloud)
├── Dockerfile                      ← Container deployment
├── .streamlit/
│   └── config.toml                 ← Theme & server settings
├── src/
│   ├── inference/
│   │   ├── pipeline.py             ← Audio preprocessing pipeline
│   │   ├── model.py                ← CBM architecture (EfficientNet-B0)
│   │   └── species_names.py        ← Species registry (182 birds)
│   └── config/
│       └── species_config.py       ← Dataset & feature constants
└── models/
    └── checkpoints/
        └── cbm_lambda0.5_epoch09.zip  ← Trained model checkpoint
```

---

## 🎼 Acoustic Concepts Explained

| Concept | Description |
|---|---|
| **Peak Frequency** | Dominant pitch of the vocalisation (Hz) |
| **Trill Rate** | Note-repetition rate (Hz) |
| **Call Duration** | Fraction of active vocalisation (0–1) |
| **FM Rate** | Frequency modulation depth (Hz/frame) |
| **Spectral Centroid** | Energy-weighted average frequency (Hz) |
| **Inter-call Silence** | Silent fraction between calls (0–1) |

These 6 concepts are the **only** signals fed to the species classifier — making every prediction
fully interpretable.

---

## 📄 License

MIT
