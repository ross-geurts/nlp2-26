import json
from pathlib import Path

# =========================
# SELECT LANGUAGE PAIR
# =========================

LANGUAGE_PAIR = "enru"   # options: ende / enes / enru

# =========================
# INPUT FILE PATHS
# =========================

DATA_DIR = Path("dev-data")

FILE_MAP = {
    "ende": "ende_dev.jsonl",
    "enes": "enes_dev.jsonl",
    "enru": "enru_dev.jsonl"
}

INPUT_FILE = DATA_DIR / FILE_MAP[LANGUAGE_PAIR]

# =========================
# OUTPUT DIRECTORY + FILE
# =========================

OUTPUT_DIR = Path("proper_terms_data")

# Create folder if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / f"{LANGUAGE_PAIR}_propterm.jsonl"

# =========================
# EXTRACT TERMINOLOGY PAIRS
# =========================

terms = set()

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        item = json.loads(line)

        proper_terms = item.get("proper_terms", {})

        for source_term, target_term in proper_terms.items():
            terms.add((source_term, target_term))

# =========================
# SAVE TERMINOLOGY PAIRS
# =========================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for source_term, target_term in sorted(terms):

        json_line = {
            "source_term": source_term,
            "target_term": target_term
        }

        f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

# =========================
# PRINT SUMMARY
# =========================

print(f"Loaded dataset: {INPUT_FILE}")
print(f"Found {len(terms)} unique terminology pairs.")
print(f"Saved terminology pairs to: {OUTPUT_FILE}")
