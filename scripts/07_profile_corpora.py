"""Profile the phishing collection, one row per source corpus.

The pct_phish column is the thing to read: if a corpus supplies only one
class, a model can 'detect phishing' on it by recognising the corpus.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RESULTS
from src.corpora import corpus_profile, load_phishing

df = load_phishing()
prof = corpus_profile(df)

print(f"total {len(df)} messages, {df['label'].mean():.1%} phishing")
print(f"dated {df['date'].notna().sum()} "
      f"({df['date'].notna().mean():.1%})\n")
print(prof.to_string(index=False))

prof.to_csv(RESULTS / "corpus_profile.csv", index=False)
print(f"\nwrote {RESULTS / 'corpus_profile.csv'}")
