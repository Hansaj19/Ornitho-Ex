"""
Ornitho-Ex | Phase 1 — Species List & Configuration
Pivoted to BirdCLEF+ 2026 dataset (Kaggle) as audio source.
No live API downloads. All audio is read-only from /kaggle/input/birdclef-plus-2026/.
Updated: 2026-09-17
"""

# ============================================================
# BIRDCLEF+ 2026 DATASET — PATH CONSTANTS (Kaggle read-only)
# Kaggle competition slug: birdclef-2026
# Confirmed mount path: /kaggle/input/competitions/birdclef-2026/
# Actual folder structure:
#   /kaggle/input/competitions/birdclef-2026/
#     ├── train_audio/<primary_label>/<primary_label>/<filename>.ogg
#             NOTE: filename in train.csv already includes the subdirectory
#             e.g. filename = "houspa/XC977863.ogg"
#             → audio_path = train_audio / filename  (no extra prefix)
#     ├── train.csv             ← metadata
#     ├── taxonomy.csv
#     ├── train_soundscapes/
#     └── sample_submission.csv
# ============================================================
BIRDCLEF_DIR        = "/kaggle/input/competitions/birdclef-2026"
BIRDCLEF_METADATA   = "/kaggle/input/competitions/birdclef-2026/train.csv"
BIRDCLEF_AUDIO_DIR  = "/kaggle/input/competitions/birdclef-2026/train_audio"

# ============================================================
# NIPS4Bplus INPUT — PATH CONSTANTS (Kaggle read-only)
# Kaggle slug: competitions/multilabel-bird-species-classification-nips2013
# Actual files visible:
#   NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS...  (CSV annotations)
#   NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_M...  (metadata)
#   NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_W...  (WAV audio files)
# ============================================================
NIPS4B_DATASET_DIR = "/kaggle/input/competitions/multilabel-bird-species-classification-nips2013"
NIPS4B_WAV_DIR     = "/kaggle/input/competitions/multilabel-bird-species-classification-nips2013"
NIPS4B_CSV_DIR     = "/kaggle/input/competitions/multilabel-bird-species-classification-nips2013"

# ============================================================
# OUTPUT — PROCESSED FEATURES (Kaggle writable)
# ============================================================
BASE_OUT_DIR        = "/kaggle/working"
PROC_FEATURES_DIR   = "/kaggle/working/processed_features"
PROC_BIRDCLEF_DIR   = "/kaggle/working/processed_features/birdclef"
PROC_NIPS4B_DIR     = "/kaggle/working/processed_features/nips4bplus/validation"

# ============================================================
# FEATURE EXTRACTION CONSTANTS
# ============================================================
# Native sample rate of BirdCLEF 2021 short audio files (ogg)
SAMPLE_RATE         = 32000   # Hz (native BirdCLEF rate)
# Alternative: set to 44100 if resampling to match NIPS4Bplus
NIPS4B_SAMPLE_RATE  = 32000   # use same rate throughout

WINDOW_DURATION_SEC = 4.0     # seconds per training segment
N_MELS              = 128     # mel frequency bins
N_FFT               = 2048    # FFT window size
HOP_LENGTH          = 512     # hop length (frames per 4s ≈ 250 frames at 32kHz)
# Expected spectrogram shape: (N_MELS, ceil(SAMPLE_RATE * WINDOW_DURATION_SEC / HOP_LENGTH))
# = (128, 250) at 32000 Hz — or (128, 345) at 44100 Hz

# Dataset balancing
MIN_RATING          = 3.5     # BirdCLEF quality threshold (0–5 star scale)
MAX_CLIPS_PER_SP    = 50      # max clips to keep per species (balanced)
TRAIN_RATIO         = 0.80    # 80% train / 10% val / 10% test
VAL_RATIO           = 0.10
TEST_RATIO          = 0.10

# ============================================================
# TRAINING SPECIES LIST
# 51 European species from NIPS4Bplus (these form the overlap
# validation subset). Extended with additional European species
# for a broader training scope (~70 total classes).
# ============================================================

# Core: NIPS4Bplus 51 European species (scientific names, 2013 taxonomy).
# These are the species for which faithfulness evaluation is possible.
NIPS4BPLUS_SPECIES = [
    "Acrocephalus arundinaceus",   # Great Reed Warbler
    "Acrocephalus scirpaceus",     # Eurasian Reed Warbler
    "Alauda arvensis",             # Eurasian Skylark
    "Anthus pratensis",            # Meadow Pipit
    "Anthus trivialis",            # Tree Pipit
    "Buteo buteo",                 # Common Buzzard
    "Carduelis carduelis",         # European Goldfinch
    "Carduelis chloris",           # European Greenfinch
    "Carduelis spinus",            # Eurasian Siskin
    "Certhia brachydactyla",       # Short-toed Treecreeper
    "Certhia familiaris",          # Eurasian Treecreeper
    "Coccothraustes coccothraustes",  # Hawfinch
    "Columba palumbus",            # Common Wood Pigeon
    "Corvus corone",               # Carrion Crow
    "Cuculus canorus",             # Common Cuckoo
    "Emberiza citrinella",         # Yellowhammer
    "Erithacus rubecula",          # European Robin
    "Ficedula hypoleuca",          # European Pied Flycatcher
    "Fringilla coelebs",           # Common Chaffinch
    "Garrulus glandarius",         # Eurasian Jay
    "Hippolais icterina",          # Icterine Warbler
    "Jynx torquilla",              # Eurasian Wryneck
    "Lanius collurio",             # Red-backed Shrike
    "Luscinia megarhynchos",       # Common Nightingale
    "Luscinia svecica",            # Bluethroat
    "Motacilla alba",              # White Wagtail
    "Motacilla flava",             # Western Yellow Wagtail
    "Muscicapa striata",           # Spotted Flycatcher
    "Oriolus oriolus",             # Eurasian Golden Oriole
    "Parus ater",                  # Coal Tit
    "Parus caeruleus",             # Eurasian Blue Tit
    "Parus major",                 # Great Tit
    "Passer domesticus",           # House Sparrow
    "Passer montanus",             # Eurasian Tree Sparrow
    "Phoenicurus ochruros",        # Black Redstart
    "Phoenicurus phoenicurus",     # Common Redstart
    "Phylloscopus collybita",      # Common Chiffchaff
    "Phylloscopus trochilus",      # Willow Warbler
    "Picus viridis",               # European Green Woodpecker
    "Regulus ignicapilla",         # Common Firecrest
    "Regulus regulus",             # Goldcrest
    "Saxicola rubetra",            # Whinchat
    "Saxicola torquata",           # European Stonechat
    "Sitta europaea",              # Eurasian Nuthatch
    "Streptopelia decaocto",       # Eurasian Collared Dove
    "Sturnus vulgaris",            # Common Starling
    "Sylvia atricapilla",          # Eurasian Blackcap
    "Sylvia borin",                # Garden Warbler
    "Sylvia communis",             # Common Whitethroat
    "Troglodytes troglodytes",     # Eurasian Wren
    "Turdus merula",               # Common Blackbird
]

# Extended training list — additional European species present in BirdCLEF 2021
EXTENDED_TRAINING_SPECIES = [
    "Turdus philomelos",           # Song Thrush
    "Turdus viscivorus",           # Mistle Thrush
    "Sylvia curruca",              # Lesser Whitethroat
    "Locustella naevia",           # Common Grasshopper Warbler
    "Acrocephalus palustris",      # Marsh Warbler
    "Emberiza schoeniclus",        # Reed Bunting
    "Parus palustris",             # Marsh Tit
    "Dendrocopos major",           # Great Spotted Woodpecker
    "Dryocopus martius",           # Black Woodpecker
    "Anthus spinoletta",           # Water Pipit
    "Oenanthe oenanthe",           # Northern Wheatear
    "Hirundo rustica",             # Barn Swallow
    "Delichon urbicum",            # Common House Martin
    "Apus apus",                   # Common Swift
    "Upupa epops",                 # Eurasian Hoopoe
    "Alcedo atthis",               # Common Kingfisher
    "Merops apiaster",             # European Bee-eater
    "Caprimulgus europaeus",       # European Nightjar
    "Sylvia nisoria",              # Barred Warbler
    "Lullula arborea",             # Wood Lark
]

# Full training list = NIPS4Bplus + extended European species
TRAINING_SPECIES = NIPS4BPLUS_SPECIES + EXTENDED_TRAINING_SPECIES

# Validation species = strictly NIPS4Bplus species ONLY
# (used ONLY for faithfulness evaluation — never mix with training scope)
VALIDATION_SPECIES = NIPS4BPLUS_SPECIES

# ============================================================
# TAXONOMY MAPPING: NIPS4Bplus (2013) → BirdCLEF 2021 labels
#
# BirdCLEF 2021 uses modern IOC/eBird taxonomy and its own
# 6-letter alpha codes as primary_label (e.g., "eurrob1").
# The train_metadata.csv also includes scientific_name column
# which uses modern taxonomy — we match against that directly.
#
# This dict maps NIPS4Bplus 2013 scientific name → modern
# scientific name used in BirdCLEF 2021 train_metadata.csv.
# Species not in this dict are already in modern taxonomy.
# ============================================================
TAXONOMY_MAP_NIPS4B_TO_BIRDCLEF = {
    # NIPS4Bplus 2013 name          : BirdCLEF 2021 scientific_name
    "Carduelis chloris"             : "Chloris chloris",
    "Carduelis spinus"              : "Spinus spinus",
    "Parus ater"                    : "Periparus ater",
    "Parus caeruleus"               : "Cyanistes caeruleus",
    "Parus palustris"               : "Poecile palustris",
    "Saxicola torquata"             : "Saxicola rubicola",
    "Sylvia communis"               : "Curruca communis",
    "Sylvia curruca"                : "Curruca curruca",
    "Sylvia nisoria"                : "Curruca nisoria",
    "Luscinia svecica"              : "Cyanecula svecica",
    "Hippolais icterina"            : "Hippolais icterina",   # unchanged — keep for completeness
    "Motacilla flava"               : "Motacilla flava",      # unchanged
    "Garrulus glandarius"           : "Garrulus glandarius",  # unchanged
}

def get_birdclef_name(nips4b_scientific_name: str) -> str:
    """
    Translate a NIPS4Bplus 2013 scientific name to the BirdCLEF 2021
    scientific_name field as it appears in train_metadata.csv.
    If no mapping exists, the name is assumed to be current and returned as-is.
    """
    return TAXONOMY_MAP_NIPS4B_TO_BIRDCLEF.get(nips4b_scientific_name, nips4b_scientific_name)


def get_all_birdclef_names(species_list: list) -> list:
    """
    Translate a list of NIPS4Bplus scientific names to their BirdCLEF 2021
    equivalents. Returns a deduplicated list preserving order.
    """
    seen = set()
    result = []
    for sp in species_list:
        modern = get_birdclef_name(sp)
        if modern not in seen:
            seen.add(modern)
            result.append(modern)
    return result


# ============================================================
# CONVENIENCE CONFIG DICT (for notebooks / scripts)
# ============================================================
CONFIG = {
    # ── BirdCLEF+ 2026 source ─────────────────────────────
    "birdclef_dir"              : BIRDCLEF_DIR,
    "birdclef_metadata"         : BIRDCLEF_METADATA,       # train.csv
    "birdclef_audio_dir"        : BIRDCLEF_AUDIO_DIR,      # train_audio/
    # NOTE: filename in train.csv already includes subdirectory prefix
    # audio_path = birdclef_audio_dir / filename  (NOT / primary_label / filename)

    # ── NIPS4Bplus source ─────────────────────────────────
    "nips4b_dataset_dir"        : NIPS4B_DATASET_DIR,
    "nips4b_wav_dir"            : NIPS4B_WAV_DIR,
    "nips4b_csv_dir"            : NIPS4B_CSV_DIR,

    # ── Output dirs ───────────────────────────────────────
    "base_out_dir"              : BASE_OUT_DIR,
    "proc_features_dir"         : PROC_FEATURES_DIR,
    "proc_birdclef_dir"         : PROC_BIRDCLEF_DIR,
    "proc_nips4b_dir"           : PROC_NIPS4B_DIR,
    "species_list_train_path"   : PROC_FEATURES_DIR + "/species_list_train.json",
    "species_list_val_path"     : PROC_FEATURES_DIR + "/species_list_validation_overlap.json",
    "label_map_path"            : PROC_FEATURES_DIR + "/label_map.json",

    # ── Feature extraction ────────────────────────────────
    "sample_rate"               : SAMPLE_RATE,
    "window_duration_sec"       : WINDOW_DURATION_SEC,
    "n_mels"                    : N_MELS,
    "n_fft"                     : N_FFT,
    "hop_length"                : HOP_LENGTH,

    # ── Dataset balancing ─────────────────────────────────
    "min_rating"                : MIN_RATING,
    "max_clips_per_species"     : MAX_CLIPS_PER_SP,
    "train_ratio"               : TRAIN_RATIO,
    "val_ratio"                 : VAL_RATIO,
    "test_ratio"                : TEST_RATIO,
}
