"""How trustworthy are the Date: headers?

Email date headers are attacker-controlled and frequently malformed. Before a
temporal split can mean anything, the implausible ones have to be counted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.corpora import load_phishing

VALID_LO, VALID_HI = pd.Timestamp("1995-01-01", tz="UTC"), \
                     pd.Timestamp("2025-01-01", tz="UTC")

df = load_phishing(dated_only=True)
d = df["date"]

ok = d.between(VALID_LO, VALID_HI)
print(f"dated messages   {d.notna().sum()}")
print(f"in [1995, 2025)  {ok.sum()} ({ok.mean():.1%})")
print(f"before 1995      {(d < VALID_LO).sum()}")
print(f"after 2025       {(d > VALID_HI).sum()}\n")

print("plausible-date span per corpus:")
sub = df[ok]
for name, g in sub.groupby("source"):
    print(f"  {name:<14} n={len(g):<6} "
          f"{g['date'].min().date()} .. {g['date'].max().date()}  "
          f"phish={g['label'].mean():.0%}")

print("\nmessages per year (plausible only):")
print(sub.groupby(sub["date"].dt.year)["label"]
         .agg(n="size", phish="mean").round(2).to_string())
