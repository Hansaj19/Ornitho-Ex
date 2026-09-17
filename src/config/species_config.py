"""
Ornitho-Ex | Phase 1A — Xeno-canto Species List & Configuration
Defines the training species list and validation overlap list.
Used by: download_xenocanto.py, preprocessing pipeline.
"""

# ============================================================
# TRAINING SPECIES LIST
# 51 European species from NIPS4Bplus (these form the overlap
# validation subset). Extend this list with additional
# Xeno-canto species (up to 100) for broader training scope.
# ============================================================

# Core: NIPS4Bplus 51 European species (scientific names)
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
    "Coccothraustes coccothraustes", # Hawfinch
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

# Extended training list — add more Xeno-canto species here
# to reach 80–100 total training classes.
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
    "Monticola saxatilis",         # Rufous-tailed Rock Thrush
    "Hirundo rustica",             # Barn Swallow
    "Delichon urbicum",            # Common House Martin
    "Apus apus",                   # Common Swift
    "Upupa epops",                 # Eurasian Hoopoe
    "Alcedo atthis",               # Common Kingfisher
    "Merops apiaster",             # European Bee-eater
    "Coracias garrulus",           # European Roller
    "Caprimulgus europaeus",       # European Nightjar
]

# Full training list = NIPS4Bplus species + extended
TRAINING_SPECIES = NIPS4BPLUS_SPECIES + EXTENDED_TRAINING_SPECIES

# Validation species = those that overlap with NIPS4Bplus
# (determined at runtime after checking Xeno-canto coverage)
VALIDATION_SPECIES = NIPS4BPLUS_SPECIES  # all NIPS4Bplus species

# ============================================================
# DOWNLOAD CONFIGURATION
# ============================================================
CONFIG = {
    # Xeno-canto quality filter (keep only A and B rated clips)
    "quality_filter": ["A", "B"],

    # Minimum clips per species to include it in training
    "min_clips_per_species": 80,

    # Target clips per species (cap downloads to keep dataset balanced, lowered to 100 to fit Kaggle 20GB limit)
    "max_clips_per_species": 100,

    # Audio segment length in seconds (3–5 sec window)
    "segment_length_sec": 4,

    # Target sample rate (matches NIPS4Bplus)
    "sample_rate": 44100,

    # Log-mel spectrogram parameters
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,

    # Kaggle paths
    "xc_audio_dir": "/kaggle/working/raw_audio/xenocanto",
    "nips4b_audio_dir": "/kaggle/working/raw_audio/nips4bplus",
    "processed_xc_dir": "/kaggle/working/xeno_canto",
    "processed_nips4b_dir": "/kaggle/working/nips4bplus",
    "species_list_train_path": "/kaggle/working/species_list_train.json",
    "species_list_val_path": "/kaggle/working/species_list_validation_overlap.json",
    "metadata_path": "/kaggle/working/metadata.csv",

    # API
    "xc_api_base": "https://xeno-canto.org/api/3/recordings",
    "api_page_size": 100,
    "max_retries": 5,
    "backoff_base_sec": 2,
}
