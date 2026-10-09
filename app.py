"""
Ornitho-Ex  |  Streamlit Web Interface
Bird species recognition from audio using an Explainable Concept Bottleneck Model (CBM).
"""

from __future__ import annotations

import io
import os
import sys
import time
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from inference.pipeline      import process_audio, SAMPLE_RATE
from inference.model         import load_cbm_checkpoint
from inference.species_names import load_label_map, get_species_info

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ornitho-Ex • Avian Bioacoustics AI",
    page_icon="🪶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Pastel Nature Design Tokens & CSS ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-main: #0c151d;
    --bg-card: rgba(19, 35, 48, 0.72);
    --bg-card-hover: rgba(25, 45, 62, 0.85);
    --border-soft: rgba(125, 211, 252, 0.18);
    --border-glow: rgba(134, 239, 172, 0.35);
    --pastel-blue: #7dd3fc;
    --pastel-green: #86efac;
    --pastel-rose: #fbcfe8;
    --pastel-lavender: #c4b5fd;
    --pastel-canary: #fef08a;
    --pastel-apricot: #fed7aa;
    --pastel-mint: #99f6e4;
    --text-title: #f8fafc;
    --text-body: #e2e8f0;
    --text-muted: #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-body);
}

h1, h2, h3, h4, .hero-title {
    font-family: 'Outfit', sans-serif;
}

/* ── Deep soothing background with subtle mist glow ── */
.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.07) 0%, transparent 45%),
                radial-gradient(circle at 85% 80%, rgba(134, 239, 172, 0.05) 0%, transparent 40%),
                linear-gradient(160deg, #0b141b 0%, #0d1a24 50%, #0a131b 100%);
    color: var(--text-body);
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(20, 42, 54, 0.85) 0%, rgba(17, 34, 48, 0.75) 50%, rgba(15, 30, 42, 0.9) 100%);
    border-radius: 24px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 2rem;
    border: 1px solid var(--border-soft);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(16px);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(125, 211, 252, 0.12) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(100deg, #bae6fd 0%, #a7f3d0 35%, #fbcfe8 70%, #fed7aa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}

.hero-sub {
    font-size: 1.05rem;
    color: #93c5fd;
    font-weight: 400;
    margin: 0;
}

.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1.1rem;
}

.hero-pill {
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.28rem 0.75rem;
    border-radius: 9999px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.09);
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Result Cards & Glassmorphism ── */
.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 18px;
    padding: 1.35rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(14px);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

.result-card:hover {
    background: var(--bg-card-hover);
    border-color: rgba(125, 211, 252, 0.38);
    transform: translateY(-2px);
    box-shadow: 0 8px 26px rgba(0, 0, 0, 0.25);
}

/* ── Top-1 Highlight Card ── */
.top-card {
    background: linear-gradient(135deg, rgba(20, 42, 58, 0.9) 0%, rgba(18, 33, 49, 0.85) 100%);
    border-radius: 20px;
    border: 1px solid rgba(125, 211, 252, 0.32);
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}

.top-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 150px; height: 150px;
    background: radial-gradient(circle at top right, rgba(125, 211, 252, 0.15), transparent 70%);
    pointer-events: none;
}

/* ── Sidebar Styling ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #091219 0%, #0c1822 100%) !important;
    border-right: 1px solid rgba(125, 211, 252, 0.12) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(125, 211, 252, 0.12) !important;
}

/* ── File Uploader & Audio Inputs ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(125, 211, 252, 0.28) !important;
    border-radius: 16px !important;
    background: rgba(19, 35, 48, 0.4) !important;
    padding: 1rem !important;
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(125, 211, 252, 0.55) !important;
    background: rgba(19, 35, 48, 0.6) !important;
}

/* ── Buttons ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #38bdf8 0%, #34d399 100%) !important;
    color: #061923 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.6rem !important;
    box-shadow: 0 4px 16px rgba(56, 189, 248, 0.28) !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"]:hover {
    transform: translateY(-1px) scale(1.01) !important;
    box-shadow: 0 6px 22px rgba(56, 189, 248, 0.42) !important;
    color: #03121a !important;
}

button[kind="secondary"] {
    background: rgba(25, 45, 62, 0.65) !important;
    border: 1px solid rgba(125, 211, 252, 0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}

button[kind="secondary"]:hover {
    border-color: rgba(125, 211, 252, 0.45) !important;
    background: rgba(25, 45, 62, 0.9) !important;
    color: #ffffff !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #7dd3fc !important;
    opacity: 0.65;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s ease !important;
}

button[data-baseweb="tab"]:hover {
    opacity: 0.9 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    opacity: 1 !important;
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* ── Badges & Tags ── */
.badge-habitat {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.82rem;
    padding: 0.3rem 0.75rem;
    border-radius: 9999px;
    background: rgba(134, 239, 172, 0.12);
    border: 1px solid rgba(134, 239, 172, 0.25);
    color: #a7f3d0;
    font-weight: 500;
}

.confidence-pill {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 0.5rem 1rem;
    border-radius: 14px;
    background: rgba(125, 211, 252, 0.1);
    border: 1px solid rgba(125, 211, 252, 0.25);
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(125, 211, 252, 0.22); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(125, 211, 252, 0.4); }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CKPT_DIR     = os.path.join(ROOT, "models", "checkpoints")
DEFAULT_CKPT = os.path.join(CKPT_DIR, "cbm_lambda0.5_epoch09.zip")

CONCEPT_LABELS = [
    "Peak Frequency (Hz)", "Trill Rate (Hz)", "Call Duration (ratio)",
    "FM Rate (Hz/frame)", "Spectral Centroid (Hz)", "Inter-call Silence (ratio)",
]

CONCEPT_DESCRIPTIONS = {
    "Peak Frequency":      "Dominant pitch of the vocalisation — distinguishes high-pitched warblers from low-pitched doves.",
    "Trill Rate":          "How quickly notes repeat — fast trills vs. slow, measured calls.",
    "Call Duration":       "Fraction of the clip occupied by active vocalisation (0 = silent, 1 = continuous).",
    "FM Rate":             "Speed of pitch change — canaries sweep rapidly, cuckoos barely deviate.",
    "Spectral Centroid":   "Energy-weighted average frequency — 'brightness' of the sound.",
    "Inter-call Silence":  "Fraction spent silent between calls — sparse callers vs. continuous singers.",
}

# ── Soothing Bird-Inspired Pastel Palette ──────────────────────────────────────
# Robin-egg Blue / Songbird Sage / Sunrise Rose / Lilac Finch / Canary Primrose / Meadow Mint
PASTEL_COLORS = ["#7dd3fc", "#86efac", "#fbcfe8", "#c4b5fd", "#fef08a", "#99f6e4"]
BG_PLOT_DARK  = "#0c151d"
BG_PLOT_CARD  = "#132330"
ACCENT_BLUE   = "#7dd3fc"
ACCENT_GREEN  = "#86efac"
ACCENT_ROSE   = "#fbcfe8"
TEXT_SLATE    = "#94a3b8"
TEXT_LIGHT    = "#f1f5f9"
GRID_COLOR    = "#1a2c3d"

# ── Model loading (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_cbm_checkpoint(ckpt_path, device=device)
    return model, device


@st.cache_resource(show_spinner=False)
def get_label_map():
    label_map_json = os.path.join(CKPT_DIR, "label_map.json")
    return load_label_map(label_map_json if os.path.exists(label_map_json) else None)


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(audio_input, model, device, label_map, top_k: int = 5):
    import torch
    import torch.nn.functional as F

    result = process_audio(audio_input)
    tensor = result["tensor"].to(device)

    with torch.no_grad():
        logits, c_pred = model(tensor)
        probs  = F.softmax(logits, dim=-1)[0].cpu().numpy()
        c_vals = c_pred[0].cpu().numpy()

    top_idx = np.argsort(probs)[::-1][:top_k]
    top_preds = [
        {"rank": i + 1, "idx": int(idx), "probability": float(probs[idx]),
         **get_species_info(int(idx), label_map)}
        for i, idx in enumerate(top_idx)
    ]

    return {
        "predictions":  top_preds,
        "concepts":     c_vals,
        "log_mel":      result["log_mel"],
        "waveform":     result["waveform"],
        "segment":      result["segment"],
        "raw_concepts": result["concepts"],
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_spectrogram(log_mel: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.4), facecolor=BG_PLOT_DARK)
    ax.set_facecolor(BG_PLOT_CARD)
    
    img = librosa.display.specshow(
        log_mel, sr=SAMPLE_RATE, hop_length=512,
        x_axis="time", y_axis="mel", ax=ax, cmap="viridis",
    )
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=ACCENT_BLUE, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_SLATE)
    cbar.outline.set_edgecolor(GRID_COLOR)

    ax.set_title("Log-Mel Spectrogram  •  Most Active 4s Window", color=TEXT_LIGHT,
                 fontsize=11, fontweight="600", pad=10)
    ax.tick_params(colors=TEXT_SLATE, labelsize=8)
    ax.xaxis.label.set_color(TEXT_SLATE)
    ax.yaxis.label.set_color(TEXT_SLATE)
    
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
        
    plt.tight_layout()
    return fig


def plot_waveform(y: np.ndarray, sr: int = SAMPLE_RATE) -> plt.Figure:
    times = np.linspace(0, len(y) / sr, len(y))
    fig, ax = plt.subplots(figsize=(10, 2.6), facecolor=BG_PLOT_DARK)
    ax.set_facecolor(BG_PLOT_CARD)
    
    ax.plot(times, y, color=ACCENT_BLUE, linewidth=0.7, alpha=0.95)
    ax.fill_between(times, y, alpha=0.15, color=ACCENT_GREEN)
    
    ax.set_xlabel("Time (seconds)", color=TEXT_SLATE, fontsize=9)
    ax.set_ylabel("Amplitude", color=TEXT_SLATE, fontsize=9)
    ax.set_title("Audio Waveform", color=TEXT_LIGHT, fontsize=11, fontweight="600", pad=8)
    ax.tick_params(colors=TEXT_SLATE, labelsize=8)
    ax.set_xlim(0, times[-1])
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)
    
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
        
    plt.tight_layout()
    return fig


def plot_concepts(c_vals: np.ndarray, raw_concepts: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), facecolor=BG_PLOT_DARK)
    
    short_labels = ["Peak\nFreq", "Trill\nRate", "Call\nDur", "FM\nRate", "Spec\nCent", "Silence"]

    ax1.set_facecolor(BG_PLOT_CARD)
    bars1 = ax1.bar(short_labels, c_vals, color=PASTEL_COLORS, alpha=0.9, edgecolor="none", width=0.52)
    ax1.set_title("Model Concept Bottleneck", color=TEXT_LIGHT, fontweight="600", fontsize=11, pad=8)
    ax1.tick_params(colors=TEXT_SLATE, labelsize=8)
    ax1.axhline(0, color=TEXT_SLATE, linewidth=0.6, alpha=0.4)
    ax1.grid(axis="y", color=GRID_COLOR, linestyle=":", alpha=0.6)
    
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax1.set_ylabel("Concept Activation", color=TEXT_SLATE, fontsize=9)

    ax2.set_facecolor(BG_PLOT_CARD)
    raw_keys = list(raw_concepts.keys())
    raw_vals = np.array([raw_concepts[k] for k in raw_keys], dtype=float)
    max_abs   = max(np.abs(raw_vals).max(), 1e-6)
    norm_disp = raw_vals / max_abs
    short_raw = [k.replace("_", "\n") for k in raw_keys]
    
    ax2.barh(short_raw[::-1], norm_disp[::-1],
             color=PASTEL_COLORS[::-1], alpha=0.9, height=0.52)
    ax2.set_title("Measured Acoustic Properties (norm)", color=TEXT_LIGHT, fontweight="600", fontsize=11, pad=8)
    ax2.tick_params(colors=TEXT_SLATE, labelsize=8)
    ax2.axvline(0, color=TEXT_SLATE, linewidth=0.6, alpha=0.4)
    ax2.grid(axis="x", color=GRID_COLOR, linestyle=":", alpha=0.6)
    
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax2.set_xlabel("Relative Magnitude", color=TEXT_SLATE, fontsize=9)

    plt.tight_layout(pad=2.2)
    return fig


# ── Defaults ──────────────────────────────────────────────────────────────────
ckpt_path = DEFAULT_CKPT
top_k     = 5

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
      <h3 style="margin: 0; color: #7dd3fc; font-weight: 700; font-size: 1.3rem;">🪶 Ornitho-Ex</h3>
      <p style="margin: 0.2rem 0 0 0; color: #94a3b8; font-size: 0.82rem;">Explainable Avian Bioacoustics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
**Ornitho-Ex** identifies bird vocalisation species using an Explainable **Concept Bottleneck Model (CBM)** built on EfficientNet-B0 and trained on BirdCLEF+ 2026 audio data.

---

#### 🌿 6 Acoustic Concepts
- 🪶 **Robin-Egg Blue** → Peak Frequency
- 🍃 **Forest Sage** → Trill Rate
- 🌸 **Feather Rose** → Call Duration
- 🪻 **Lilac Finch** → FM Rate
- 🌼 **Canary Sun** → Spectral Centroid
- 🌱 **Meadow Mint** → Inter-call Silence

---
    """)
    st.caption("Ornitho-Ex v1.2 · CBM λ=0.5 · 182 European Species")


# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🪶 Ornitho-Ex</p>
  <p class="hero-sub">Explainable Bioacoustic Bird Sound Classification & Acoustic Concept Extraction</p>
  <div class="hero-pills">
    <span class="hero-pill" style="color: #7dd3fc;">🪶 182 Species</span>
    <span class="hero-pill" style="color: #86efac;">🌱 6 Acoustic Concepts</span>
    <span class="hero-pill" style="color: #fbcfe8;">🧬 Bottleneck Explainability</span>
    <span class="hero-pill" style="color: #c4b5fd;">🎙️ Real-time Audio AI</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Audio Input Section ───────────────────────────────────────────────────────
tab_upload, tab_record = st.tabs(["📁  Upload Audio File", "🎙️  Record Live Audio"])
audio_bytes: bytes | None = None
audio_name: str = ""

with tab_upload:
    st.markdown("##### 📤 Drop a bird call recording")
    st.caption("Supported formats: WAV, MP3, OGG, FLAC, M4A · 4–30s field recordings recommended")
    uploaded = st.file_uploader(
        "Drop your audio file here",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        audio_bytes = uploaded.read()
        audio_name  = uploaded.name
        st.audio(audio_bytes, format=uploaded.type)
        st.markdown(f"""
        <div style="padding:0.5rem 0.8rem; border-radius:10px; background:rgba(134,239,172,0.12); border:1px solid rgba(134,239,172,0.3); color:#86efac; font-size:0.88rem; display:inline-block; margin-top:0.4rem;">
          ✓ Ready to analyse: <strong>{audio_name}</strong> ({len(audio_bytes)/1024:.1f} KB)
        </div>
        """, unsafe_allow_html=True)

with tab_record:
    st.markdown("##### 🎙️ Record sound via microphone")
    st.caption("Click the red circle to record bird song, then stop when finished.")
    recorded = st.audio_input("Microphone input", label_visibility="collapsed", key="mic_recorder")
    if recorded is not None:
        audio_bytes = recorded.read()
        audio_name  = "live_recording.wav"
        st.markdown("""
        <div style="padding:0.5rem 0.8rem; border-radius:10px; background:rgba(125,211,252,0.12); border:1px solid rgba(125,211,252,0.3); color:#7dd3fc; font-size:0.88rem; display:inline-block; margin-top:0.4rem;">
          ✓ Audio captured! Click <strong>Analyse Call</strong> below.
        </div>
        """, unsafe_allow_html=True)


# ── Analyse Action ────────────────────────────────────────────────────
st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
run_col, _ = st.columns([1.2, 4.8])
with run_col:
    analyse_btn = st.button(
        "🔬 Analyse Call",
        type="primary",
        use_container_width=True,
        disabled=(audio_bytes is None),
    )

if audio_bytes and analyse_btn:
    with st.spinner("🧠 Initialising Concept Bottleneck Model…"):
        model, device = load_model(ckpt_path)
        label_map     = get_label_map()

    with st.spinner("🔊 Processing audio features & calculating concept activations…"):
        t0      = time.perf_counter()
        output  = run_inference(audio_bytes, model, device, label_map, top_k=top_k)
        elapsed = time.perf_counter() - t0

    st.markdown(f"""
    <div style="margin: 1.2rem 0; padding: 0.6rem 1rem; border-radius: 12px; background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.25); color: #7dd3fc; font-size: 0.9rem; display: flex; align-items: center; justify-content: space-between;">
      <span>✓ Identification finished in <strong>{elapsed:.2f}s</strong></span>
      <span style="font-size: 0.8rem; color: #94a3b8;">Compute: <code>{device.upper()}</code></span>
    </div>
    """, unsafe_allow_html=True)

    predictions  = output["predictions"]
    c_vals       = output["concepts"]
    log_mel      = output["log_mel"]
    waveform     = output["waveform"]
    segment      = output["segment"]
    raw_concepts = output["raw_concepts"]

    # ── Top-1 Highlight Card ──────────────────────────────────────────────────
    top1 = predictions[0]
    st.markdown(f"""
    <div class="top-card">
      <div style="display:flex;align-items:center;gap:1.4rem;flex-wrap:wrap;">
        <span style="font-size:4rem;filter:drop-shadow(0 4px 12px rgba(0,0,0,0.4));">{top1['emoji']}</span>
        <div style="flex:1;min-width:240px;">
          <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.2rem;">
            <span style="font-size:0.75rem;font-weight:700;letter-spacing:0.08em;color:#7dd3fc;background:rgba(125,211,252,0.15);padding:0.2rem 0.6rem;border-radius:6px;">TOP MATCH</span>
          </div>
          <h2 style="margin:0;font-size:2.2rem;font-weight:800;color:#f8fafc;letter-spacing:-0.01em;">{top1['common']}</h2>
          <p style="margin:0.15rem 0 0 0;font-size:1.05rem;color:#7dd3fc;font-style:italic;">{top1['scientific']}</p>
          <p style="margin:0.5rem 0 0 0;font-size:0.92rem;color:#cbd5e1;line-height:1.5;">{top1['description']}</p>
        </div>
        <div class="confidence-pill" style="min-width:110px;">
          <span style="font-size:2.4rem;font-weight:800;color:#86efac;line-height:1;">{top1['probability']*100:.1f}%</span>
          <span style="font-size:0.72rem;color:#94a3b8;font-weight:600;letter-spacing:0.08em;margin-top:0.3rem;">CONFIDENCE</span>
        </div>
      </div>
      <div style="margin-top:1.2rem;padding-top:1rem;border-top:1px solid rgba(125,211,252,0.15);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.6rem;">
        <span class="badge-habitat">🏔️ {top1['habitat']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Predictions + Acoustic Side-by-Side ────────────────────────────────────
    col_pred, col_acou = st.columns([1.15, 1], gap="large")

    with col_pred:
        st.markdown("#### 🏆 Top Ranked Species")
        for pred in predictions:
            pct = pred["probability"] * 100
            bar_color = "#7dd3fc" if pred["rank"] == 1 else "#86efac" if pred["rank"] == 2 else "#c4b5fd" if pred["rank"] == 3 else "#94a3b8"
            st.markdown(f"""
            <div class="result-card" style="padding:0.9rem 1.2rem;margin-bottom:0.75rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-size:0.98rem;font-weight:600;color:#f8fafc;display:flex;align-items:center;gap:0.5rem;">
                  <span style="font-size:1.2em;">{pred['emoji']}</span>
                  <span>{pred['common']}</span>
                </span>
                <span style="font-size:0.98rem;font-weight:700;color:{bar_color};">{pct:.1f}%</span>
              </div>
              <div style="background:rgba(255,255,255,0.06);border-radius:9999px;height:7px;overflow:hidden;">
                <div style="background:linear-gradient(90deg, {bar_color}cc, {bar_color});width:{max(pct, 1.5):.1f}%;height:100%;border-radius:9999px;"></div>
              </div>
              <p style="margin:4px 0 0 0;font-size:0.75rem;color:#94a3b8;font-style:italic;">{pred['scientific']}</p>
            </div>
            """, unsafe_allow_html=True)

    with col_acou:
        st.markdown("#### 🎼 Extracted Acoustic Concepts")
        acou_items = [
            ("Peak Frequency",     f"{raw_concepts['peak_frequency']:.0f} Hz",      "#7dd3fc", "🧭"),
            ("Trill Rate",         f"{raw_concepts['trill_rate']:.2f} Hz",           "#86efac", "🌿"),
            ("Call Duration",      f"{raw_concepts['call_duration']*100:.1f}%",      "#fbcfe8", "🌸"),
            ("FM Rate",            f"{raw_concepts['fm_rate']:.1f} Hz/frame",        "#c4b5fd", "🪻"),
            ("Spectral Centroid",  f"{raw_concepts['spectral_centroid']:.0f} Hz",    "#fef08a", "🌼"),
            ("Inter-call Silence", f"{raw_concepts['inter_call_silence']*100:.1f}%", "#99f6e4", "🌱"),
        ]
        for label, val, color, icon in acou_items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:0.65rem 1.1rem;margin-bottom:0.55rem;
                        background:rgba(19,35,48,0.7);border-radius:12px;
                        border:1px solid rgba(125,211,252,0.12);border-left:3.5px solid {color};">
              <span style="font-size:0.88rem;color:#cbd5e1;display:flex;align-items:center;gap:0.4rem;">
                <span>{icon}</span> {label}
              </span>
              <span style="font-size:0.95rem;font-weight:700;color:{color};">{val}</span>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("ℹ️ Learn about these acoustic concepts"):
            for k, desc in CONCEPT_DESCRIPTIONS.items():
                st.markdown(f"**{k}** — {desc}")

    # ── Visualisations ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Explainability & Audio Visualisations")
    viz1, viz2, viz3 = st.tabs(["🌈 Log-Mel Spectrogram", "〰️ Audio Waveform", "🧬 CBM Concept Activations"])

    with viz1:
        st.pyplot(plot_spectrogram(log_mel), use_container_width=True)
        st.caption("Optimal 4-second audio window selected via energy detection for maximum vocalisation clarity.")

    with viz2:
        st.pyplot(plot_waveform(waveform), use_container_width=True)
        st.caption(f"Full audio clip waveform — {len(waveform)/SAMPLE_RATE:.2f}s duration at {SAMPLE_RATE} Hz mono.")

    with viz3:
        st.pyplot(plot_concepts(c_vals, raw_concepts), use_container_width=True)
        st.info(
            "**Left Plot:** Latent concept activations learned by the CBM bottleneck — the explicit features feeding into species classification. "
            "**Right Plot:** Directly extracted bioacoustic measurements computed directly from the audio signal."
        )

    # ── Export Results ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💾 Export Analysis Data")
    dl1, dl2, dl3 = st.columns(3)

    report = {
        "audio_file":       audio_name,
        "inference_time_s": round(elapsed, 3),
        "device":           device,
        "top_predictions": [
            {"rank": p["rank"], "common_name": p["common"],
             "scientific_name": p["scientific"],
             "confidence_pct": round(p["probability"]*100, 2),
             "habitat": p["habitat"]}
            for p in predictions
        ],
        "acoustic_concepts": {k: round(float(v), 4) for k, v in raw_concepts.items()},
    }
    with dl1:
        st.download_button("📄 Download JSON Report", json.dumps(report, indent=2),
                           "ornitho_ex_report.json", "application/json",
                           use_container_width=True)

    buf_spec = io.BytesIO()
    plot_spectrogram(log_mel).savefig(buf_spec, format="png", dpi=160,
                                      bbox_inches="tight", facecolor=BG_PLOT_DARK)
    buf_spec.seek(0)
    with dl2:
        st.download_button("🖼️ Save Spectrogram PNG", buf_spec, "spectrogram.png",
                           "image/png", use_container_width=True)

    buf_wav = io.BytesIO()
    sf.write(buf_wav, segment, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf_wav.seek(0)
    with dl3:
        st.download_button("🔊 Save Active Clip WAV", buf_wav, "bird_call_segment.wav",
                           "audio/wav", use_container_width=True)

elif not audio_bytes:
    st.markdown("""
    <div style="text-align:center;padding:4.5rem 2rem 3rem 2rem;background:rgba(19,35,48,0.3);border-radius:24px;border:1px dashed rgba(125,211,252,0.2);margin-top:1.5rem;">
      <p style="font-size:4.5rem;margin-bottom:1rem;filter:drop-shadow(0 6px 16px rgba(0,0,0,0.3));">🪶</p>
      <h3 style="font-size:1.6rem;font-weight:700;
                background:linear-gradient(100deg,#bae6fd 0%,#a7f3d0 50%,#fbcfe8 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.6rem;">
        Upload or Record a Bird Call
      </h3>
      <p style="font-size:0.95rem;max-width:560px;margin:0 auto;color:#94a3b8;line-height:1.65;">
        Ornitho-Ex automatically isolates the most prominent vocalisation segment, identifies the species with confidence metrics, and visualises six interpretable acoustic concepts.
      </p>
      <div style="display:flex;justify-content:center;gap:1.2rem;margin-top:2rem;flex-wrap:wrap;">
        <span style="font-size:0.85rem;color:#7dd3fc;background:rgba(125,211,252,0.08);padding:0.4rem 0.9rem;border-radius:9999px;border:1px solid rgba(125,211,252,0.18);">
          🎙️ High-accuracy acoustic CBM
        </span>
        <span style="font-size:0.85rem;color:#86efac;background:rgba(134,239,172,0.08);padding:0.4rem 0.9rem;border-radius:9999px;border:1px solid rgba(134,239,172,0.18);">
          🌿 European Bird Species
        </span>
        <span style="font-size:0.85rem;color:#fbcfe8;background:rgba(251,207,232,0.08);padding:0.4rem 0.9rem;border-radius:9999px;border:1px solid rgba(251,207,232,0.18);">
          🧬 Explainable XAI Latents
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)
