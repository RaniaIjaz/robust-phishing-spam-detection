"""Descriptive profile of the corpus - class balance, lengths, vocabulary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import load_sms_spam

df = load_sms_spam()
df["chars"] = df["text"].str.len()
df["words"] = df["text"].str.split().str.len()
df["has_digit"] = df["text"].str.contains(r"\d")
df["has_url"] = df["text"].str.contains(r"http|www\.", case=False)
df["upper_frac"] = df["text"].apply(
    lambda t: sum(c.isupper() for c in t) / max(len(t), 1))

print(f"messages     {len(df)}")
print(f"spam         {df['label'].sum()} ({df['label'].mean():.1%})")
print(f"exact dupes  {df['text'].duplicated().sum()}")
print(f"vocabulary   {len(set(w.lower() for t in df['text'] for w in t.split()))}")

print("\nby class:")
g = df.groupby("label").agg(
    n=("text", "size"),
    med_chars=("chars", "median"),
    med_words=("words", "median"),
    pct_digit=("has_digit", "mean"),
    pct_url=("has_url", "mean"),
    upper=("upper_frac", "mean"),
).round(3)
g.index = ["ham", "spam"]
print(g.to_string())
