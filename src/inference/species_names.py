"""
Ornitho-Ex | Species Name and Info Registry

Maps numeric class indices (0-181) to species information.
"""

from __future__ import annotations
import json
from pathlib import Path

_SPECIES_DB: dict[str, dict] = {
    "Acrocephalus arundinaceus": {
        "common": "Great Reed Warbler",
        "emoji": "🎵",
        "habitat": "Reed beds & wetlands",
        "description": "Europe's largest warbler, famous for its loud, grating song from tall reeds.",
        "color": "#4ade80"
    },
    "Acrocephalus palustris": {
        "common": "Marsh Warbler",
        "emoji": "🌿",
        "habitat": "Damp meadows & marshes",
        "description": "Exceptional mimic — can imitate over 200 other bird species.",
        "color": "#34d399"
    },
    "Acrocephalus scirpaceus": {
        "common": "Eurasian Reed Warbler",
        "emoji": "🪺",
        "habitat": "Reed beds",
        "description": "A secretive, brown warbler producing a rhythmic churring song.",
        "color": "#6ee7b7"
    },
    "Alauda arvensis": {
        "common": "Eurasian Skylark",
        "emoji": "🌤️",
        "habitat": "Open farmland & grassland",
        "description": "Celebrated for its continuous, soaring song delivered high in the sky.",
        "color": "#fde68a"
    },
    "Alcedo atthis": {
        "common": "Common Kingfisher",
        "emoji": "💎",
        "habitat": "Clear rivers & streams",
        "description": "A jewel-like bird with brilliant blue-orange plumage, diving for fish.",
        "color": "#38bdf8"
    },
    "Anthus pratensis": {
        "common": "Meadow Pipit",
        "emoji": "🌾",
        "habitat": "Open moorland & grassland",
        "description": "A small, streaky pipit with a parachuting song-flight display.",
        "color": "#a3e635"
    },
    "Anthus spinoletta": {
        "common": "Water Pipit",
        "emoji": "💧",
        "habitat": "Mountain streams & coasts",
        "description": "Breeds at high altitudes, descending to coasts and waterways in winter.",
        "color": "#7dd3fc"
    },
    "Anthus trivialis": {
        "common": "Tree Pipit",
        "emoji": "🌳",
        "habitat": "Open woodland & heathland",
        "description": "Performs a parachuting song-flight from a tree perch.",
        "color": "#86efac"
    },
    "Apus apus": {
        "common": "Common Swift",
        "emoji": "⚡",
        "habitat": "Open skies & urban areas",
        "description": "Spends almost its entire life airborne; screaming parties around rooftops.",
        "color": "#c084fc"
    },
    "Buteo buteo": {
        "common": "Common Buzzard",
        "emoji": "🦅",
        "habitat": "Woodland edges & farmland",
        "description": "Europe's most common large raptor, soaring on broad wings with a mewing call.",
        "color": "#f59e0b"
    },
    "Caprimulgus europaeus": {
        "common": "European Nightjar",
        "emoji": "🌙",
        "habitat": "Heathland & open woodland",
        "description": "A cryptic nocturnal bird known for its mechanical churring song at dusk.",
        "color": "#818cf8"
    },
    "Carduelis carduelis": {
        "common": "European Goldfinch",
        "emoji": "✨",
        "habitat": "Gardens & weedy fields",
        "description": "A brilliantly coloured finch with a liquid, tinkling song.",
        "color": "#fb923c"
    },
    "Certhia brachydactyla": {
        "common": "Short-toed Treecreeper",
        "emoji": "🌲",
        "habitat": "Deciduous woodland",
        "description": "Spirals up tree trunks probing bark for insects.",
        "color": "#a8a29e"
    },
    "Certhia familiaris": {
        "common": "Eurasian Treecreeper",
        "emoji": "🌲",
        "habitat": "Coniferous & mixed woodland",
        "description": "Mouse-like bird that spirals up bark using its stiff tail for support.",
        "color": "#78716c"
    },
    "Chloris chloris": {
        "common": "European Greenfinch",
        "emoji": "🍃",
        "habitat": "Gardens & woodland edges",
        "description": "A stout green finch with a distinctive wheezing call.",
        "color": "#4ade80"
    },
    "Coccothraustes coccothraustes": {
        "common": "Hawfinch",
        "emoji": "🪨",
        "habitat": "Mature deciduous woodland",
        "description": "Massive bill can crack cherry stones; shy and often heard before seen.",
        "color": "#f97316"
    },
    "Columba palumbus": {
        "common": "Common Wood Pigeon",
        "emoji": "🕊️",
        "habitat": "Woodland & farmland",
        "description": "Europe's largest pigeon; characteristic five-note cooing.",
        "color": "#94a3b8"
    },
    "Corvus corone": {
        "common": "Carrion Crow",
        "emoji": "🖤",
        "habitat": "Farmland & urban areas",
        "description": "An all-black, intelligent corvid with a harsh, cawing call.",
        "color": "#374151"
    },
    "Cuculus canorus": {
        "common": "Common Cuckoo",
        "emoji": "🕐",
        "habitat": "Woodland & open country",
        "description": "Famous brood parasite with the iconic two-note 'cu-ckoo' call.",
        "color": "#60a5fa"
    },
    "Curruca communis": {
        "common": "Common Whitethroat",
        "emoji": "🌼",
        "habitat": "Scrub & hedgerows",
        "description": "Scratchy, energetic song delivered from the top of a bramble.",
        "color": "#fef3c7"
    },
    "Curruca curruca": {
        "common": "Lesser Whitethroat",
        "emoji": "🌸",
        "habitat": "Scrub & thickets",
        "description": "Delivers a rattling song from dense cover; grey with a dark mask.",
        "color": "#e2e8f0"
    },
    "Curruca nisoria": {
        "common": "Barred Warbler",
        "emoji": "🔵",
        "habitat": "Bushy scrub",
        "description": "Large, robust warbler with barred underparts and bright yellow eyes.",
        "color": "#dbeafe"
    },
    "Cyanecula svecica": {
        "common": "Bluethroat",
        "emoji": "💙",
        "habitat": "Willowherb & scrub near water",
        "description": "A stunning blue, orange and white breast patch on an otherwise robin-like bird.",
        "color": "#3b82f6"
    },
    "Cyanistes caeruleus": {
        "common": "Eurasian Blue Tit",
        "emoji": "💙",
        "habitat": "Deciduous woods & garden feeders",
        "description": "Energetic garden tit with bright cobalt cap and yellow chest.",
        "color": "#7dd3fc"
    },
    "Delichon urbicum": {
        "common": "Common House Martin",
        "emoji": "🏠",
        "habitat": "Urban & suburban areas",
        "description": "Builds mud-cup nests under eaves; white rump distinctive in flight.",
        "color": "#e2e8f0"
    },
    "Dendrocopos major": {
        "common": "Great Spotted Woodpecker",
        "emoji": "🎯",
        "habitat": "Woodland",
        "description": "Unmistakable black-and-white pied woodpecker with loud drumming.",
        "color": "#ef4444"
    },
    "Dryocopus martius": {
        "common": "Black Woodpecker",
        "emoji": "⚫",
        "habitat": "Mature coniferous forest",
        "description": "Europe's largest woodpecker; crow-sized, all-black with red crown.",
        "color": "#1f2937"
    },
    "Emberiza citrinella": {
        "common": "Yellowhammer",
        "emoji": "☀️",
        "habitat": "Farmland & hedgerows",
        "description": "The 'little-bit-of-bread-and-no-cheese' song is instantly recognisable.",
        "color": "#fbbf24"
    },
    "Emberiza schoeniclus": {
        "common": "Reed Bunting",
        "emoji": "🌊",
        "habitat": "Reed beds & damp grassland",
        "description": "Male has a striking black head and white collar in summer.",
        "color": "#64748b"
    },
    "Erithacus rubecula": {
        "common": "European Robin",
        "emoji": "❤️",
        "habitat": "Woodland & gardens",
        "description": "Britain's unofficial national bird; melodious year-round singer.",
        "color": "#ef4444"
    },
    "Ficedula hypoleuca": {
        "common": "European Pied Flycatcher",
        "emoji": "⚫",
        "habitat": "Deciduous woodland",
        "description": "Snaps insects in mid-air; males striking black and white in summer.",
        "color": "#334155"
    },
    "Fringilla coelebs": {
        "common": "Common Chaffinch",
        "emoji": "🍂",
        "habitat": "Woodland, parks & gardens",
        "description": "One of Europe's most abundant birds; the male has a pink breast.",
        "color": "#f97316"
    },
    "Garrulus glandarius": {
        "common": "Eurasian Jay",
        "emoji": "💜",
        "habitat": "Deciduous woodland",
        "description": "Colourful crow relative with a loud, screaming alarm call.",
        "color": "#a855f7"
    },
    "Hippolais icterina": {
        "common": "Icterine Warbler",
        "emoji": "🎼",
        "habitat": "Open woodland & gardens",
        "description": "A loud, fast-singing warbler with a yellow-green tinge.",
        "color": "#84cc16"
    },
    "Hirundo rustica": {
        "common": "Barn Swallow",
        "emoji": "🌍",
        "habitat": "Open farmland & villages",
        "description": "Long forked tail and metallic blue back; the quintessential summer visitor.",
        "color": "#2563eb"
    },
    "Jynx torquilla": {
        "common": "Eurasian Wryneck",
        "emoji": "🐍",
        "habitat": "Open woodland & orchards",
        "description": "An atypical woodpecker that twists its head snake-like when alarmed.",
        "color": "#a16207"
    },
    "Lanius collurio": {
        "common": "Red-backed Shrike",
        "emoji": "🔴",
        "habitat": "Thorny scrub & heathland",
        "description": "Impales prey on thorns to create a 'larder'; male has russet back.",
        "color": "#dc2626"
    },
    "Locustella naevia": {
        "common": "Common Grasshopper Warbler",
        "emoji": "🦗",
        "habitat": "Dense grassland & marsh",
        "description": "Produces an insect-like reeling song — often mistaken for a grasshopper.",
        "color": "#a3e635"
    },
    "Lullula arborea": {
        "common": "Wood Lark",
        "emoji": "🎶",
        "habitat": "Heathland & open woodland",
        "description": "A beautiful, fluty song delivered in a circular song-flight.",
        "color": "#fcd34d"
    },
    "Luscinia megarhynchos": {
        "common": "Common Nightingale",
        "emoji": "🌙",
        "habitat": "Dense undergrowth",
        "description": "Arguably Europe's finest songster, singing loudly even after dark.",
        "color": "#f472b6"
    },
    "Merops apiaster": {
        "common": "European Bee-eater",
        "emoji": "🌈",
        "habitat": "Open country with sandy banks",
        "description": "Spectacularly colourful; catches bees and removes their stings before eating.",
        "color": "#f59e0b"
    },
    "Motacilla alba": {
        "common": "White Wagtail",
        "emoji": "⬛",
        "habitat": "Open areas near water",
        "description": "Constantly wags its long tail; familiar on rooftops and roads.",
        "color": "#e5e7eb"
    },
    "Motacilla flava": {
        "common": "Western Yellow Wagtail",
        "emoji": "💛",
        "habitat": "Wet meadows & river edges",
        "description": "Bright yellow below; follows cattle to catch disturbed insects.",
        "color": "#eab308"
    },
    "Muscicapa striata": {
        "common": "Spotted Flycatcher",
        "emoji": "🌫️",
        "habitat": "Woodland edges & gardens",
        "description": "Grey-brown flycatcher that sallies out after insects from a perch.",
        "color": "#9ca3af"
    },
    "Oenanthe oenanthe": {
        "common": "Northern Wheatear",
        "emoji": "🏔️",
        "habitat": "Open moorland & stony ground",
        "description": "White rump flashing as it bounds across rocky ground.",
        "color": "#d97706"
    },
    "Oriolus oriolus": {
        "common": "Eurasian Golden Oriole",
        "emoji": "🌟",
        "habitat": "Mature deciduous woodland",
        "description": "Striking yellow and black male with a rich, fluting whistle.",
        "color": "#fbbf24"
    },
    "Parus major": {
        "common": "Great Tit",
        "emoji": "🎨",
        "habitat": "Woodland & gardens",
        "description": "One of the most adaptable European birds with a huge repertoire.",
        "color": "#84cc16"
    },
    "Passer domesticus": {
        "common": "House Sparrow",
        "emoji": "🏙️",
        "habitat": "Urban & suburban areas",
        "description": "The quintessential 'little brown bird'; closely associated with humans.",
        "color": "#a16207"
    },
    "Passer montanus": {
        "common": "Eurasian Tree Sparrow",
        "emoji": "🌳",
        "habitat": "Rural areas & woodland edge",
        "description": "Distinguished from House Sparrow by chestnut cap and black ear patch.",
        "color": "#92400e"
    },
    "Periparus ater": {
        "common": "Coal Tit",
        "emoji": "🌲",
        "habitat": "Conifer plantations & gardens",
        "description": "Small tit with white nape patch foraging high in spruce needles.",
        "color": "#94a3b8"
    },
    "Phoenicurus ochruros": {
        "common": "Black Redstart",
        "emoji": "🔥",
        "habitat": "Rocky ground & urban areas",
        "description": "Dark, robin-like bird with a conspicuous orange-red tail.",
        "color": "#f97316"
    },
    "Phoenicurus phoenicurus": {
        "common": "Common Redstart",
        "emoji": "🧡",
        "habitat": "Open woodland",
        "description": "Striking male with orange-red tail; a summer visitor to Europe.",
        "color": "#fb923c"
    },
    "Phylloscopus collybita": {
        "common": "Common Chiffchaff",
        "emoji": "📢",
        "habitat": "Woodland & scrub",
        "description": "Named after its monotonous two-note 'chiff-chaff' song.",
        "color": "#86efac"
    },
    "Phylloscopus trochilus": {
        "common": "Willow Warbler",
        "emoji": "🎵",
        "habitat": "Open woodland & scrub",
        "description": "Beautiful descending cascading song; Europe's commonest warbler.",
        "color": "#a3e635"
    },
    "Picus viridis": {
        "common": "European Green Woodpecker",
        "emoji": "💚",
        "habitat": "Open woodland & parkland",
        "description": "Loud 'yaffle' laughing call; specialises in eating ants.",
        "color": "#22c55e"
    },
    "Poecile palustris": {
        "common": "Marsh Tit",
        "emoji": "🌿",
        "habitat": "Damp woodland",
        "description": "Very similar to Willow Tit but with a glossy black cap.",
        "color": "#713f12"
    },
    "Regulus ignicapilla": {
        "common": "Common Firecrest",
        "emoji": "🔥",
        "habitat": "Mixed and coniferous woodland",
        "description": "Europe's smallest bird; orange crown stripe, white supercilium.",
        "color": "#f97316"
    },
    "Regulus regulus": {
        "common": "Goldcrest",
        "emoji": "👑",
        "habitat": "Coniferous woodland",
        "description": "Europe's smallest bird; the jewel-like yellow-orange crown stripe.",
        "color": "#fde047"
    },
    "Saxicola rubicola": {
        "common": "European Stonechat",
        "emoji": "🪨",
        "habitat": "Heathland & gorse",
        "description": "Perches prominently on gorse; hard 'tac-tac' call like stones knocking.",
        "color": "#dc2626"
    },
    "Saxicola rubetra": {
        "common": "Whinchat",
        "emoji": "🌄",
        "habitat": "Upland grassland & heath",
        "description": "Summer visitor; male has orange breast and bold white supercilium.",
        "color": "#f59e0b"
    },
    "Sitta europaea": {
        "common": "Eurasian Nuthatch",
        "emoji": "🔵",
        "habitat": "Deciduous woodland",
        "description": "The only bird that can descend trees head-first; loud, piping calls.",
        "color": "#3b82f6"
    },
    "Spinus spinus": {
        "common": "Eurasian Siskin",
        "emoji": "🌱",
        "habitat": "Coniferous woodland",
        "description": "Small, streaky green finch; gregarious on alder and birch catkins.",
        "color": "#65a30d"
    },
    "Streptopelia decaocto": {
        "common": "Eurasian Collared Dove",
        "emoji": "🕊️",
        "habitat": "Urban & suburban areas",
        "description": "Rapidly colonised Europe in the 20th century; gentle cooing call.",
        "color": "#d4b483"
    },
    "Sturnus vulgaris": {
        "common": "Common Starling",
        "emoji": "⭐",
        "habitat": "Farmland, parks & urban",
        "description": "A superb mimic; forms spectacular murmurations at dusk.",
        "color": "#7c3aed"
    },
    "Sylvia atricapilla": {
        "common": "Eurasian Blackcap",
        "emoji": "🎩",
        "habitat": "Woodland & gardens",
        "description": "Rich, varied song rivals the Nightingale; male has a black cap.",
        "color": "#1f2937"
    },
    "Sylvia borin": {
        "common": "Garden Warbler",
        "emoji": "🌺",
        "habitat": "Dense woodland undergrowth",
        "description": "A plain, nondescript bird with a beautiful, sustained warbling song.",
        "color": "#86efac"
    },
    "Troglodytes troglodytes": {
        "common": "Eurasian Wren",
        "emoji": "🏡",
        "habitat": "Dense undergrowth & woodland",
        "description": "Tiny but incredibly loud; one of Britain's commonest birds.",
        "color": "#92400e"
    },
    "Turdus merula": {
        "common": "Common Blackbird",
        "emoji": "🎵",
        "habitat": "Woodland, parks & gardens",
        "description": "The quintessential British songster with a rich, fluting melody.",
        "color": "#1f2937"
    },
    "Turdus philomelos": {
        "common": "Song Thrush",
        "emoji": "🎶",
        "habitat": "Woodland & gardens",
        "description": "Repeats each musical phrase two or three times; uses stones to crack snails.",
        "color": "#a16207"
    },
    "Turdus viscivorus": {
        "common": "Mistle Thrush",
        "emoji": "🌧️",
        "habitat": "Open woodland & parks",
        "description": "The 'storm-cock'; sings defiantly from treetops in wild weather.",
        "color": "#6b7280"
    },
    "Upupa epops": {
        "common": "Eurasian Hoopoe",
        "emoji": "👑",
        "habitat": "Open country with short grass",
        "description": "Unmistakable cinnamon plumage and striking fan crest; 'oop-oop-oop' call.",
        "color": "#f97316"
    },
    "Accipiter gentilis": {
        "common": "Northern Goshawk",
        "emoji": "🦅",
        "habitat": "Dense conifer & mixed forest",
        "description": "Fierce apex raptor of mature old-growth woodland.",
        "color": "#7dd3fc"
    },
    "Accipiter nisus": {
        "common": "Eurasian Sparrowhawk",
        "emoji": "🦅",
        "habitat": "Woodland, gardens & farmland",
        "description": "Agile raptor hunting songbirds through forest clearings.",
        "color": "#94a3b8"
    },
    "Acrocephalus schoenobaenus": {
        "common": "Sedge Warbler",
        "emoji": "🌾",
        "habitat": "Reedbeds & wet ditches",
        "description": "Lively, improvisational singer from marshes with rapid bursts.",
        "color": "#86efac"
    },
    "Aegithalos caudatus": {
        "common": "Long-tailed Tit",
        "emoji": "🍡",
        "habitat": "Hedgerows & deciduous scrub",
        "description": "Tiny fluffy ball with disproportionately long tail, moves in flocks.",
        "color": "#fbcfe8"
    },
    "Aegolius funereus": {
        "common": "Boreal Owl",
        "emoji": "🦉",
        "habitat": "Taiga & mountain spruce forests",
        "description": "Small nocturnal owl of northern boreal forests with bell-like calls.",
        "color": "#c4b5fd"
    },
    "Anas crecca": {
        "common": "Eurasian Teal",
        "emoji": "🦆",
        "habitat": "Shallow wetlands & marshes",
        "description": "Smallest European dabbling duck with a piping whistle.",
        "color": "#7dd3fc"
    },
    "Anas platyrhynchos": {
        "common": "Mallard",
        "emoji": "🦆",
        "habitat": "Ponds, lakes & rivers",
        "description": "Familiar iridescent green-headed wild duck.",
        "color": "#86efac"
    },
    "Anser anser": {
        "common": "Greylag Goose",
        "emoji": "🪿",
        "habitat": "Lakes, estuaries & grazing marshes",
        "description": "Ancestor of domestic geese with resonant honking calls.",
        "color": "#fed7aa"
    },
    "Anthus cervinus": {
        "common": "Red-throated Pipit",
        "emoji": "🌿",
        "habitat": "Tundra & damp coastal marshes",
        "description": "Subtle pinkish throat in breeding plumage, high shrill calls.",
        "color": "#fef08a"
    },
    "Aquila chrysaetos": {
        "common": "Golden Eagle",
        "emoji": "👑",
        "habitat": "Mountain crags & moorland",
        "description": "Majestic apex predator soaring over remote highland ridges.",
        "color": "#fed7aa"
    },
    "Ardea cinerea": {
        "common": "Grey Heron",
        "emoji": "🪶",
        "habitat": "Rivers, lakes & tidal mudflats",
        "description": "Statuesque wading bird stalking fish with lightning strikes.",
        "color": "#94a3b8"
    },
    "Asio otus": {
        "common": "Long-eared Owl",
        "emoji": "🦉",
        "habitat": "Conifer belts near open fields",
        "description": "Secretive nocturnal owl with prominent head tufts and orange eyes.",
        "color": "#fed7aa"
    },
    "Athene noctua": {
        "common": "Little Owl",
        "emoji": "🦉",
        "habitat": "Orchards, old farmland & stone walls",
        "description": "Compact daylight-active owl with piercing yellow eyes.",
        "color": "#fef08a"
    },
    "Aythya fuligula": {
        "common": "Tufted Duck",
        "emoji": "🦆",
        "habitat": "Reservoirs, lakes & park waters",
        "description": "Diving duck with bold white flank patches and drooping crest.",
        "color": "#7dd3fc"
    },
    "Bombycilla garrulus": {
        "common": "Bohemian Waxwing",
        "emoji": "🍒",
        "habitat": "Boreal forests & berry trees",
        "description": "Crested winter visitor famous for sleek plumage and trilling calls.",
        "color": "#fbcfe8"
    },
    "Botaurus stellaris": {
        "common": "Eurasian Bittern",
        "emoji": "🎋",
        "habitat": "Dense reedbeds",
        "description": "Master of camouflage producing a deep resonant foghorn boom.",
        "color": "#fed7aa"
    },
    "Bubo bubo": {
        "common": "Eurasian Eagle-Owl",
        "emoji": "🦉",
        "habitat": "Rocky cliffs & wooded valleys",
        "description": "One of the largest owls in the world with deep booming hoot.",
        "color": "#fed7aa"
    },
    "Buteo lagopus": {
        "common": "Rough-legged Buzzard",
        "emoji": "🦅",
        "habitat": "Arctic tundra & open winter plains",
        "description": "Hovering northern raptor with feathered legs and pale tail.",
        "color": "#94a3b8"
    },
    "Calcarius lapponicus": {
        "common": "Lapland Longspur",
        "emoji": "❄️",
        "habitat": "Arctic tundra & coastal saltmarshes",
        "description": "Hardy northern songbird with chestnut nape and musical flight call.",
        "color": "#fef08a"
    },
    "Calidris alpina": {
        "common": "Dunlin",
        "emoji": "🌊",
        "habitat": "Coastal mudflats & peat bogs",
        "description": "Small shorebird flocking in synchronised waves along tides.",
        "color": "#7dd3fc"
    },
    "Carduelis flammea": {
        "common": "Common Redpoll",
        "emoji": "🍓",
        "habitat": "Birch scrub & boreal woodland",
        "description": "Acrobatic finch with crimson forehead foraging on birch seeds.",
        "color": "#fbcfe8"
    },
    "Carpodacus erythrinus": {
        "common": "Common Rosefinch",
        "emoji": "🌸",
        "habitat": "River valleys & willow thickets",
        "description": "Crimson-red male singing a sweet flute-like song.",
        "color": "#fbcfe8"
    },
    "Charadrius dubius": {
        "common": "Little Ringed Plover",
        "emoji": "🏖️",
        "habitat": "Gravel pits & shingle riverbeds",
        "description": "Active shorebird with distinctive golden eye-ring.",
        "color": "#fef08a"
    },
    "Charadrius hiaticula": {
        "common": "Common Ringed Plover",
        "emoji": "🏖️",
        "habitat": "Sandy & pebble beaches",
        "description": "Bold black-and-white neck collar and piping melodic call.",
        "color": "#fed7aa"
    },
    "Chlidonias niger": {
        "common": "Black Tern",
        "emoji": "🪶",
        "habitat": "Inland marshes & shallow fens",
        "description": "Graceful marsh tern dipping to pluck insects from water.",
        "color": "#7dd3fc"
    },
    "Cinclus cinclus": {
        "common": "White-throated Dipper",
        "emoji": "🌊",
        "habitat": "Fast-flowing rocky streams",
        "description": "Songbird that walks underwater along riverbeds for nymphs.",
        "color": "#94a3b8"
    },
    "Circus aeruginosus": {
        "common": "Western Marsh Harrier",
        "emoji": "🌾",
        "habitat": "Expansive reedbeds & fens",
        "description": "Broad-winged harrier quartering low over reedbeds on V-shaped wings.",
        "color": "#fed7aa"
    },
    "Circus cyaneus": {
        "common": "Hen Harrier",
        "emoji": "🦅",
        "habitat": "Heather moorland & open marsh",
        "description": "Ghost-grey male performing tumbling courtship sky-dances.",
        "color": "#94a3b8"
    },
    "Clangula hyemalis": {
        "common": "Long-tailed Duck",
        "emoji": "❄️",
        "habitat": "Arctic tundra lakes & coastal seas",
        "description": "Deep-diving sea duck with musical yodelling call.",
        "color": "#7dd3fc"
    },
    "Columba oenas": {
        "common": "Stock Dove",
        "emoji": "🕊️",
        "habitat": "Old woodland & parkland",
        "description": "Fast-flying wild pigeon with iridescent green neck patch.",
        "color": "#c4b5fd"
    },
    "Coracias garrulus": {
        "common": "European Roller",
        "emoji": "💎",
        "habitat": "Warm open lowland & oak savannah",
        "description": "Brilliant turquoise bird of prey doing aerobatic rolling displays.",
        "color": "#7dd3fc"
    },
    "Corvus corax": {
        "common": "Northern Raven",
        "emoji": "🖤",
        "habitat": "Mountains, sea cliffs & deep forest",
        "description": "Massive intelligent corvid with deep cronking call and diamond tail.",
        "color": "#94a3b8"
    },
    "Corvus frugilegus": {
        "common": "Rook",
        "emoji": "🌳",
        "habitat": "Farmland & communal rookeries",
        "description": "Sociable corvid with bare greyish-white bill base.",
        "color": "#94a3b8"
    },
    "Corvus monedula": {
        "common": "Western Jackdaw",
        "emoji": "👀",
        "habitat": "Old buildings, cliffs & farmland",
        "description": "Compact corvid with silvery nape and piercing pale eyes.",
        "color": "#94a3b8"
    },
    "Coturnix coturnix": {
        "common": "Common Quail",
        "emoji": "🌾",
        "habitat": "Cereal fields & dry grassland",
        "description": "Secretive ground-dwelling gamebird with rhythmic piping call.",
        "color": "#fed7aa"
    },
    "Crex crex": {
        "common": "Corn Crake",
        "emoji": "🌾",
        "habitat": "Hay meadows & tall herb fields",
        "description": "Nocturnal rasping call across traditional hay meadows.",
        "color": "#fed7aa"
    },
    "Cygnus cygnus": {
        "common": "Whooper Swan",
        "emoji": "🪿",
        "habitat": "Remote taiga bogs & open floodlands",
        "description": "Large wild swan with lemon-yellow wedge on bill and bugling call.",
        "color": "#fef08a"
    },
    "Cygnus olor": {
        "common": "Mute Swan",
        "emoji": "🦢",
        "habitat": "Lowland lakes, canals & rivers",
        "description": "Graceful classic swan with orange bill and black basal knob.",
        "color": "#f8fafc"
    },
    "Dendrocopos minor": {
        "common": "Lesser Spotted Woodpecker",
        "emoji": "🪵",
        "habitat": "Deciduous woodland canopies & orchards",
        "description": "Sparrow-sized woodpecker foraging in high rotten twigs.",
        "color": "#fbcfe8"
    },
    "Emberiza calandra": {
        "common": "Corn Bunting",
        "emoji": "🌾",
        "habitat": "Arable farmland & open plains",
        "description": "Chunky bunting singing a jingling keys song from fence posts.",
        "color": "#fed7aa"
    },
    "Emberiza cirlus": {
        "common": "Cirl Bunting",
        "emoji": "🌱",
        "habitat": "Sunlit hillsides, hedges & vineyards",
        "description": "Striking black-and-yellow facial pattern with buzzing trill.",
        "color": "#fef08a"
    },
    "Emberiza hortulana": {
        "common": "Ortolan Bunting",
        "emoji": "🌾",
        "habitat": "Dry farmland with scattered trees",
        "description": "Olive-grey head, pink bill, and sweet melancholic cadence.",
        "color": "#fed7aa"
    },
    "Eremophila alpestris": {
        "common": "Horned Lark",
        "emoji": "🏔️",
        "habitat": "Alpine meadows & arctic tundra",
        "description": "Yellow face mask with tiny black feather horns.",
        "color": "#fef08a"
    },
    "Falco columbarius": {
        "common": "Merlin",
        "emoji": "🦅",
        "habitat": "Open moorland & coastal marshes",
        "description": "Europe's smallest falcon, fast and low-flying hunter.",
        "color": "#94a3b8"
    },
    "Falco peregrinus": {
        "common": "Peregrine Falcon",
        "emoji": "⚡",
        "habitat": "High sea cliffs & gorge crags",
        "description": "Fastest animal on Earth, stooping at over 300 km/h.",
        "color": "#7dd3fc"
    },
    "Falco subbuteo": {
        "common": "Eurasian Hobby",
        "emoji": "🦅",
        "habitat": "Heathland & open woodland edges",
        "description": "Sleek falcon catching dragonflies and swifts in mid-air.",
        "color": "#c4b5fd"
    },
    "Falco tinnunculus": {
        "common": "Common Kestrel",
        "emoji": "🦅",
        "habitat": "Roadside verges, farmland & coasts",
        "description": "Iconic raptor hovering stationary in wind hunting voles.",
        "color": "#fed7aa"
    },
    "Ficedula albicollis": {
        "common": "Collared Flycatcher",
        "emoji": "🤍",
        "habitat": "Mature broadleaf oak forests",
        "description": "Bold black and white plumage with unbroken white collar ring.",
        "color": "#f8fafc"
    },
    "Ficedula parva": {
        "common": "Red-breasted Flycatcher",
        "emoji": "🧡",
        "habitat": "Old-growth beech & spruce forest",
        "description": "Tiny flycatcher with Robin-like orange throat patch.",
        "color": "#fed7aa"
    },
    "Fulica atra": {
        "common": "Eurasian Coot",
        "emoji": "🪶",
        "habitat": "Open lakes, fens & park ponds",
        "description": "Slate-black waterbird with stark white bill and frontal shield.",
        "color": "#94a3b8"
    },
    "Gallinago gallinago": {
        "common": "Common Snipe",
        "emoji": "🌾",
        "habitat": "Bogs, wet meadows & marshy fens",
        "description": "Long-billed wader creating drumming sound with tail feathers.",
        "color": "#fed7aa"
    },
    "Gallinula chloropus": {
        "common": "Common Moorhen",
        "emoji": "🪶",
        "habitat": "Ponds, ditches & canal margins",
        "description": "Red-and-yellow bill with white flank stripe and bobbing head.",
        "color": "#fbcfe8"
    },
    "Gavia arctica": {
        "common": "Black-throated Loon",
        "emoji": "🌊",
        "habitat": "Deep northern boreal lakes",
        "description": "Striking geometric chequered plumage and haunting wailing cries.",
        "color": "#7dd3fc"
    },
    "Gavia stellata": {
        "common": "Red-throated Loon",
        "emoji": "🌊",
        "habitat": "Small tundra tarns & coastal bays",
        "description": "Slender uptilted bill with goose-like flight calls.",
        "color": "#7dd3fc"
    },
    "Glaucidium passerinum": {
        "common": "Eurasian Pygmy Owl",
        "emoji": "🦉",
        "habitat": "Old conifer & mixed mountain forest",
        "description": "Smallest European owl, no larger than a starling, active at dawn.",
        "color": "#c4b5fd"
    },
    "Grus grus": {
        "common": "Common Crane",
        "emoji": "🪶",
        "habitat": "Peat bogs, fens & shallow lakes",
        "description": "Tall elegant wader performing leap dances with bugling trumpets.",
        "color": "#94a3b8"
    },
    "Gyps fulvus": {
        "common": "Griffon Vulture",
        "emoji": "🦅",
        "habitat": "Gorges, mountain cliffs & open steppe",
        "description": "Huge broad-winged scavenger riding thermal updrafts over canyons.",
        "color": "#fed7aa"
    },
    "Haematopus ostralegus": {
        "common": "Eurasian Oystercatcher",
        "emoji": "🏖️",
        "habitat": "Rocky shores & sandy estuaries",
        "description": "Bold pied plumage, crimson carrot bill, and loud piping calls.",
        "color": "#fbcfe8"
    },
    "Haliaeetus albicilla": {
        "common": "White-tailed Eagle",
        "emoji": "🦅",
        "habitat": "Sea coasts, fjords & large lakes",
        "description": "Europe's largest eagle with massive broad wings.",
        "color": "#fed7aa"
    },
    "Hippolais polyglotta": {
        "common": "Melodious Warbler",
        "emoji": "🎵",
        "habitat": "Sunny scrub, hedges & open wood edges",
        "description": "Warm-climate warbler with rapid, tumbling melodic song.",
        "color": "#fef08a"
    },
    "Lagopus lagopus": {
        "common": "Willow Ptarmigan",
        "emoji": "❄️",
        "habitat": "Heather moorland & birch tundra",
        "description": "Arctic grouse turning pure snowy white in winter.",
        "color": "#f8fafc"
    },
    "Lagopus muta": {
        "common": "Rock Ptarmigan",
        "emoji": "🏔️",
        "habitat": "High alpine boulder fields & tundra",
        "description": "Hardy alpine dweller surviving extreme mountain blizzards.",
        "color": "#f8fafc"
    },
    "Lanius excubitor": {
        "common": "Great Grey Shrike",
        "emoji": "🦅",
        "habitat": "Heaths, bogs & rough pasture",
        "description": "Songbird raptor impaling prey on thorns; stark bandit mask.",
        "color": "#94a3b8"
    },
    "Larus argentatus": {
        "common": "European Herring Gull",
        "emoji": "🌊",
        "habitat": "Coasts, cliffs & seaside harbours",
        "description": "Classic large coastal gull with pale grey back and loud laughing calls.",
        "color": "#7dd3fc"
    },
    "Larus canus": {
        "common": "Mew Gull",
        "emoji": "🌊",
        "habitat": "Lakes, coasts & moorland bogs",
        "description": "Gentle-faced small gull with greenish-yellow legs.",
        "color": "#7dd3fc"
    },
    "Larus ridibundus": {
        "common": "Black-headed Gull",
        "emoji": "🌊",
        "habitat": "Inland marshes, lakes & coasts",
        "description": "Chocolate-brown summer hood and loud raucous calls in colonies.",
        "color": "#94a3b8"
    },
    "Limosa limosa": {
        "common": "Black-tailed Godwit",
        "emoji": "🌾",
        "habitat": "Wet coastal grassland & estuaries",
        "description": "Long straight bill and graceful aerial display over meadows.",
        "color": "#fed7aa"
    },
    "Linaria cannabina": {
        "common": "Common Linnet",
        "emoji": "🌸",
        "habitat": "Gorse heath, scrub & farm hedges",
        "description": "Crimson-breasted seed-eater singing melodious twittering songs.",
        "color": "#fbcfe8"
    },
    "Linaria flavirostris": {
        "common": "Twite",
        "emoji": "🌾",
        "habitat": "Upland moorland & coastal saltmarshes",
        "description": "Tough mountain finch with warm buff throat and pinkish rump.",
        "color": "#fed7aa"
    },
    "Locustella luscinioides": {
        "common": "Savi's Warbler",
        "emoji": "🎋",
        "habitat": "Dense reedbeds & fens",
        "description": "Reeling continuous mechanical song like an electric reel.",
        "color": "#86efac"
    },
    "Loxia curvirostra": {
        "common": "Red Crossbill",
        "emoji": "🌲",
        "habitat": "Conifer forests & pine plantations",
        "description": "Specialised crossed mandibles for prising conifer pine cones.",
        "color": "#fbcfe8"
    },
    "Lymnocryptes minimus": {
        "common": "Jack Snipe",
        "emoji": "🌾",
        "habitat": "Bogs & marshy tussocks",
        "description": "Secretive wader with distinctive bobbing motion when feeding.",
        "color": "#fed7aa"
    },
    "Mergus merganser": {
        "common": "Common Merganser",
        "emoji": "🦆",
        "habitat": "Clear forested rivers & lakes",
        "description": "Sleek sawbill duck diving for fish in clear mountain streams.",
        "color": "#7dd3fc"
    },
    "Mergus serrator": {
        "common": "Red-breasted Merganser",
        "emoji": "🦆",
        "habitat": "Coastal bays, fjords & estuaries",
        "description": "Shaggy double crest and slender serrated red bill.",
        "color": "#7dd3fc"
    },
    "Milvus migrans": {
        "common": "Black Kite",
        "emoji": "🦅",
        "habitat": "Rivers, lakes & open countryside",
        "description": "Abundant agile raptor with shallow-forked tail soaring over waters.",
        "color": "#fed7aa"
    },
    "Milvus milvus": {
        "common": "Red Kite",
        "emoji": "🪁",
        "habitat": "Wooded farmland & valley slopes",
        "description": "Magnificent deeply forked reddish tail and buoyant flight.",
        "color": "#fed7aa"
    },
    "Monticola saxatilis": {
        "common": "Common Rock Thrush",
        "emoji": "🏔️",
        "habitat": "Sunlit alpine scree & rocky slopes",
        "description": "Cobalt blue head and bright fiery-orange underparts.",
        "color": "#7dd3fc"
    },
    "Monticola solitarius": {
        "common": "Blue Rock Thrush",
        "emoji": "🏔️",
        "habitat": "Sea cliffs, gorges & stone ruins",
        "description": "All-dark blue plumage with clear ringing flute song.",
        "color": "#7dd3fc"
    },
    "Motacilla cinerea": {
        "common": "Grey Wagtail",
        "emoji": "💧",
        "habitat": "Fast rushing mountain streams",
        "description": "Sulphur-yellow underbelly and exceptionally long bobbing tail.",
        "color": "#fef08a"
    },
    "Nucifraga caryocatactes": {
        "common": "Spotted Nutcracker",
        "emoji": "🌲",
        "habitat": "Subalpine conifer & stone pine woods",
        "description": "Stout-billed crow caching thousands of hazel and pine nuts.",
        "color": "#fed7aa"
    },
    "Numenius arquata": {
        "common": "Eurasian Curlew",
        "emoji": "🌾",
        "habitat": "Upland moorland & coastal mudflats",
        "description": "Europe's largest wader with long downcurved bill and bubbling cry.",
        "color": "#fed7aa"
    },
    "Numenius phaeopus": {
        "common": "Whimbrel",
        "emoji": "🌾",
        "habitat": "Tundra, heaths & migration coasts",
        "description": "Striped crown pattern and rapid rippling seven-whistle call.",
        "color": "#fed7aa"
    },
    "Nycticorax nycticorax": {
        "common": "Black-crowned Night Heron",
        "emoji": "🌙",
        "habitat": "Freshwater lagoons, rivers & swamps",
        "description": "Stocky nocturnal heron with ruby-red eyes and long white nape plumes.",
        "color": "#94a3b8"
    },
    "Otus scops": {
        "common": "Eurasian Scops Owl",
        "emoji": "🦉",
        "habitat": "Mediterranean gardens, parks & olive groves",
        "description": "Tiny camouflaged owl repeating a clear single-note musical whistle.",
        "color": "#86efac"
    },
    "Pandion haliaetus": {
        "common": "Osprey",
        "emoji": "🦅",
        "habitat": "Clean lakes, coastal bays & reservoirs",
        "description": "Specialised fish-hunting raptor plunging feet-first into water.",
        "color": "#7dd3fc"
    },
    "Panurus biarmicus": {
        "common": "Bearded Reedling",
        "emoji": "🎋",
        "habitat": "Large unbroken reedbeds",
        "description": "Cinnamon-tawny reed bird with drooping black moustache and pinging calls.",
        "color": "#fed7aa"
    },
    "Perdix perdix": {
        "common": "Grey Partridge",
        "emoji": "🌾",
        "habitat": "Traditional farmland & arable stubble",
        "description": "Classic farmland covey bird with horseshoe-shaped belly patch.",
        "color": "#fed7aa"
    },
    "Pernis apivorus": {
        "common": "European Honey Buzzard",
        "emoji": "🐝",
        "habitat": "Deciduous woodland with clearings",
        "description": "Summer migrant raptor specialising in digging out wasp nests.",
        "color": "#fed7aa"
    },
    "Phalacrocorax carbo": {
        "common": "Great Cormorant",
        "emoji": "🌊",
        "habitat": "Coasts, estuaries & large inland lakes",
        "description": "Large dark waterbird perching with wings outstretched to dry.",
        "color": "#94a3b8"
    },
    "Phasianus colchicus": {
        "common": "Ring-necked Pheasant",
        "emoji": "🪶",
        "habitat": "Woodland edges & agricultural fields",
        "description": "Long-tailed colourful gamebird with explosive wing-clap and crow.",
        "color": "#fed7aa"
    },
    "Philomachus pugnax": {
        "common": "Ruff",
        "emoji": "🪶",
        "habitat": "Wet meadows & marshy shorelines",
        "description": "Spectacular varied breeding ruffs in male communal lek dances.",
        "color": "#fbcfe8"
    },
    "Phylloscopus bonelli": {
        "common": "Western Bonelli's Warbler",
        "emoji": "🌿",
        "habitat": "Dry sunny hillside oak & pine woods",
        "description": "Silvery-white underparts and clear monotonous spinning trill.",
        "color": "#86efac"
    },
    "Phylloscopus inornatus": {
        "common": "Yellow-browed Warbler",
        "emoji": "🍃",
        "habitat": "Taiga birch scrub & coastal bushes",
        "description": "Hyperactive eastern migrant with prominent double wingbars.",
        "color": "#fef08a"
    },
    "Phylloscopus sibilatrix": {
        "common": "Wood Warbler",
        "emoji": "🍃",
        "habitat": "Mature beech & oak closed canopy",
        "description": "Bright lemon-yellow throat with accelerating spinning-coin song.",
        "color": "#fef08a"
    },
    "Pica pica": {
        "common": "Eurasian Magpie",
        "emoji": "🪶",
        "habitat": "Farmland, parks, gardens & scrub",
        "description": "Clever, inquisitive corvid with long iridescent metallic tail.",
        "color": "#7dd3fc"
    },
    "Picoides tridactylus": {
        "common": "Eurasian Three-toed Woodpecker",
        "emoji": "🪵",
        "habitat": "Old boreal taiga & dead spruce trees",
        "description": "Northern woodpecker with yellow crown and three toes.",
        "color": "#fef08a"
    },
    "Picus canus": {
        "common": "Grey-headed Woodpecker",
        "emoji": "🪵",
        "habitat": "Deciduous river valleys & hills",
        "description": "Grey neck with scarlet crown patch and descending piping call.",
        "color": "#86efac"
    },
    "Platalea leucorodia": {
        "common": "Eurasian Spoonbill",
        "emoji": "🪶",
        "habitat": "Shallow fens, lagoons & coastal flats",
        "description": "Pure white plumage with remarkable spatula-shaped foraging bill.",
        "color": "#f8fafc"
    },
    "Plectrophenax nivalis": {
        "common": "Snow Bunting",
        "emoji": "❄️",
        "habitat": "High mountain tops & arctic tundra",
        "description": "Snowflake bird flourishing on frozen arctic windswept ridges.",
        "color": "#f8fafc"
    },
    "Pluvialis apricaria": {
        "common": "European Golden Plover",
        "emoji": "✨",
        "habitat": "Upland peat moorland & winter fields",
        "description": "Spangled golden back with haunting melancholic piping whistle.",
        "color": "#fef08a"
    },
    "Pluvialis squatarola": {
        "common": "Grey Plover",
        "emoji": "🌊",
        "habitat": "Tide-swept coastal mudflats & beaches",
        "description": "Silver-chequered shorebird with prominent black armpit patches.",
        "color": "#94a3b8"
    },
    "Podiceps cristatus": {
        "common": "Great Crested Grebe",
        "emoji": "🪶",
        "habitat": "Large freshwater lakes & reservoirs",
        "description": "Spectacular head-shaking courtship weed dances on open waters.",
        "color": "#fbcfe8"
    },
    "Porzana porzana": {
        "common": "Spotted Crake",
        "emoji": "🎋",
        "habitat": "Dense swampy vegetation & flooded fens",
        "description": "Nocturnal whiplash call echoing across flooded fens.",
        "color": "#fed7aa"
    },
    "Prunella modularis": {
        "common": "Dunnock",
        "emoji": "🍂",
        "habitat": "Garden hedges, scrub & woodland edge",
        "description": "Unassuming hedge bird with fine warbler bill and rapid warble.",
        "color": "#94a3b8"
    },
    "Pyrrhocorax graculus": {
        "common": "Alpine Chough",
        "emoji": "🏔️",
        "habitat": "High alpine peaks & ski crags",
        "description": "Yellow-billed acrobat gliding along cliff faces in mountain gales.",
        "color": "#fef08a"
    },
    "Pyrrhocorax pyrrhocorax": {
        "common": "Red-billed Chough",
        "emoji": "🏔️",
        "habitat": "Sea cliffs & mountain pastures",
        "description": "Curved coral-red bill with buoyant aerial loops and ringing cry.",
        "color": "#fbcfe8"
    },
    "Pyrrhula pyrrhula": {
        "common": "Eurasian Bullfinch",
        "emoji": "🍓",
        "habitat": "Dense woodland, orchard buds & scrub",
        "description": "Glowing rose-red breast and soft melancholic piping whistle.",
        "color": "#fbcfe8"
    },
    "Rallus aquaticus": {
        "common": "Water Rail",
        "emoji": "🎋",
        "habitat": "Dense wetlands, reedbeds & marshes",
        "description": "Secretive wader with long red bill and bizarre piglet-like squeals.",
        "color": "#fbcfe8"
    },
    "Recurvirostra avosetta": {
        "common": "Pied Avocet",
        "emoji": "🏖️",
        "habitat": "Coastal brackish lagoons & fens",
        "description": "Striking upturned bill sweeping sideways through shallow water.",
        "color": "#7dd3fc"
    },
    "Remiz pendulinus": {
        "common": "Eurasian Penduline Tit",
        "emoji": "🪺",
        "habitat": "Willows & reeds along lake shores",
        "description": "Master builder of intricate hanging pouch nests from willow down.",
        "color": "#fed7aa"
    },
    "Riparia riparia": {
        "common": "Sand Martin",
        "emoji": "🏖️",
        "habitat": "Sandy riverbanks & gravel quarry cliffs",
        "description": "Tiny agile brown-backed swallow nesting in colonial sand burrows.",
        "color": "#fed7aa"
    },
    "Scolopax rusticola": {
        "common": "Eurasian Woodcock",
        "emoji": "🍂",
        "habitat": "Damp deciduous & mixed forest floor",
        "description": "Nocturnal roding display flight at dusk with croaking grunt calls.",
        "color": "#fed7aa"
    },
    "Serinus serinus": {
        "common": "European Serin",
        "emoji": "🌻",
        "habitat": "Sunny parkland, vineyards & gardens",
        "description": "Tiny bright yellow finch with sizzling, rapid song like crushed glass.",
        "color": "#fef08a"
    },
    "Somateria mollissima": {
        "common": "Common Eider",
        "emoji": "🌊",
        "habitat": "Rocky sea coasts & coastal skerries",
        "description": "Heavy sea duck producing soft coos on ocean swells.",
        "color": "#7dd3fc"
    },
    "Sterna hirundo": {
        "common": "Common Tern",
        "emoji": "🌊",
        "habitat": "Coastal islands, shingle beaches & lakes",
        "description": "Graceful sea swallow with forked streamers plunging for minnows.",
        "color": "#7dd3fc"
    },
    "Streptopelia turtur": {
        "common": "European Turtle Dove",
        "emoji": "🕊️",
        "habitat": "Sunny woodland edge & farm hedgerows",
        "description": "Gentle purring turr-turr song of summer woodland borders.",
        "color": "#fbcfe8"
    },
    "Strix aluco": {
        "common": "Tawny Owl",
        "emoji": "🦉",
        "habitat": "Deciduous woods, parkland & farm groves",
        "description": "Classic hooting owl of European nights.",
        "color": "#fed7aa"
    },
    "Strix uralensis": {
        "common": "Ural Owl",
        "emoji": "🦉",
        "habitat": "Old-growth taiga & mixed boreal woods",
        "description": "Large, pale grey forest owl with long tail and fierce nest defense.",
        "color": "#94a3b8"
    },
    "Sylvia melanocephala": {
        "common": "Sardinian Warbler",
        "emoji": "🌿",
        "habitat": "Mediterranean maquis & coastal scrub",
        "description": "Velvet-black hood with fiery red eye-ring and churring alarm rattle.",
        "color": "#fbcfe8"
    },
    "Sylvia undata": {
        "common": "Dartford Warbler",
        "emoji": "🪻",
        "habitat": "Lowland heather heath & gorse commons",
        "description": "Dark wine-red underparts with long cocked tail in dense gorse.",
        "color": "#c4b5fd"
    },
    "Tachybaptus ruficollis": {
        "common": "Little Grebe",
        "emoji": "💧",
        "habitat": "Vegetated ponds, quiet oxbows & ditches",
        "description": "Tiny dabchick diving with rapid bubbling whinnying trill.",
        "color": "#86efac"
    },
    "Tadorna tadorna": {
        "common": "Common Shelduck",
        "emoji": "🦆",
        "habitat": "Estuaries, mudflats & sandy coasts",
        "description": "Goose-like coastal duck with chestnut chest band and red bill knob.",
        "color": "#fed7aa"
    },
    "Tetrao urogallus": {
        "common": "Western Capercaillie",
        "emoji": "🌲",
        "habitat": "Ancient Caledonian & taiga pine forest",
        "description": "Giant woodland grouse performing clicking and cork-popping lek song.",
        "color": "#7dd3fc"
    },
    "Tichodroma muraria": {
        "common": "Wallcreeper",
        "emoji": "🦋",
        "habitat": "Vertical alpine limestone cliff faces",
        "description": "Crimson-winged butterfly-like bird hopping up sheer precipices.",
        "color": "#fbcfe8"
    },
    "Tringa glareola": {
        "common": "Wood Sandpiper",
        "emoji": "🌿",
        "habitat": "Taiga bogs & marshy pool margins",
        "description": "Spotted wader performing high yodelling song flights over muskeg.",
        "color": "#86efac"
    },
    "Tringa totanus": {
        "common": "Common Redshank",
        "emoji": "🏖️",
        "habitat": "Saltmarshes, wet coastal meadows & fens",
        "description": "Sentinel of the marsh with bright orange-red legs and ringing alarm.",
        "color": "#fed7aa"
    },
    "Turdus iliacus": {
        "common": "Redwing",
        "emoji": "🍂",
        "habitat": "Birch woods & winter berry hedges",
        "description": "Winter thrush with bold creamy eyestripe and rusty flank patches.",
        "color": "#fbcfe8"
    },
    "Turdus pilaris": {
        "common": "Fieldfare",
        "emoji": "🍁",
        "habitat": "Open birch woodland & orchards",
        "description": "Handsome blue-grey capped thrush calling with chattering notes.",
        "color": "#94a3b8"
    },
    "Turdus torquatus": {
        "common": "Ring Ouzel",
        "emoji": "🏔️",
        "habitat": "Rocky heather slopes & alpine gullies",
        "description": "Mountain blackbird sporting a crisp white crescent breast gorget.",
        "color": "#94a3b8"
    },
    "Tyto alba": {
        "common": "Western Barn Owl",
        "emoji": "🤍",
        "habitat": "Farmland, barns & rough grass margins",
        "description": "Silent ghostly hunter with heart-shaped white face and rasping shriek.",
        "color": "#f8fafc"
    },
    "Vanellus vanellus": {
        "common": "Northern Lapwing",
        "emoji": "🌾",
        "habitat": "Pastures, arable wetlands & moorland",
        "description": "Iridescent green plumage, slender upcurved crest, and peewit cry.",
        "color": "#7dd3fc"
    }
}

def _build_fallback_index_map() -> dict[int, dict]:
    all_sci = sorted(_SPECIES_DB.keys())
    return {i: {"scientific": sp, **_SPECIES_DB[sp]} for i, sp in enumerate(all_sci)}

def load_label_map(label_map_path: str | None = None) -> dict[int, dict]:
    if label_map_path:
        p = Path(label_map_path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            idx2label = raw.get("idx2label", {})
            result = {}
            for idx_str, sci_name in idx2label.items():
                idx = int(idx_str)
                info = _SPECIES_DB.get(sci_name, {})
                result[idx] = {
                    "scientific":   sci_name,
                    "common":       info.get("common",      sci_name),
                    "emoji":        info.get("emoji",       "🐦"),
                    "habitat":      info.get("habitat",     "Temperate Habitats"),
                    "description":  info.get("description", "Identified avian species."),
                    "color":        info.get("color",       "#7dd3fc"),
                }
            return result
    return _build_fallback_index_map()

def get_species_info(idx: int, label_map: dict) -> dict:
    return label_map.get(idx, {
        "scientific":  f"Avian Species #{idx}",
        "common":      f"Species #{idx}",
        "emoji":       "🐦",
        "habitat":     "Temperate Habitats",
        "description": f"Model identified vocalisation patterns for class {idx}.",
        "color":       "#7dd3fc",
    })
