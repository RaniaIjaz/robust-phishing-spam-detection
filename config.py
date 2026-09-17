"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

for _d in (DATA_RAW, DATA_PROCESSED, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.25

# Near-duplicate clustering: cosine similarity on char 5-grams above this
# threshold means two messages are treated as the same underlying template.
DUPLICATE_THRESHOLD = 0.80

# A missed phishing message is assumed this many times more costly than a
# false alarm. Used for threshold selection in evaluate.py.
FN_COST_RATIO = 10.0
