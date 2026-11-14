MODELS = {
    "qwen": "Qwen/Qwen3-4B-Instruct-2507",
    "gemma": "google/gemma-3-4b-it",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}

TOPICS = [
    "war and armed conflict",
    "economic and fiscal crises",
    "politics and diplomacy issues",
    "emerging health crises",
    "natural disaster and ecological disaster",
    "social conflicts and religious discrimination",
]

REGIONS = ["asia", "europe", "africa", "americas", "oceania"]

# ground-truth ids -> label names
CONCEPT_MAPS = {
    "Topic": {str(i + 1): t for i, t in enumerate(TOPICS)},
    "Region": {str(i + 1): r for i, r in enumerate(REGIONS)},
}

MAX_WORDS = 250
TEMPERATURE = 0.75
MAX_NEW_TOKENS = {"xcot": 1000, "cross_tot": 1500}
