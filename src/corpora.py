"""The Kaggle phishing-email collection, loaded per source corpus.

The archive also ships `phishing_email.csv`, which is all six corpora already
concatenated with provenance discarded. That file is deliberately not used
here: once you cannot tell which corpus a message came from, you cannot test
whether a classifier is detecting phishing or merely detecting Enron.

Loading the sources separately keeps `source` as a column, which is what makes
leave_one_corpus_out() possible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, SEED

PHISHING_DIR = DATA_RAW / "phishing"

# Files with a `date` column can take part in temporal evaluation.
CORPORA = {
    "ceas":         ("CEAS_08.csv",        True),
    "nazario":      ("Nazario.csv",        True),
    "nigerian":     ("Nigerian_Fraud.csv", True),
    "spamassassin": ("SpamAssasin.csv",    True),   # sic: one 's' in the file
    "enron":        ("Enron.csv",          False),
    "ling":         ("Ling.csv",           False),
}

DATED = [k for k, (_, d) in CORPORA.items() if d]


def load_phishing(sources: list[str] | None = None,
                  dated_only: bool = False) -> pd.DataFrame:
    """Load per-corpus CSVs into DataFrame[text, label, source, date].

    `text` is subject and body joined, matching how a filter would see the
    message. `date` is NaT for corpora that do not carry one.
    """
    names = sources or (DATED if dated_only else list(CORPORA))
    frames = []

    for name in names:
        fname, has_date = CORPORA[name]
        path = PHISHING_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python scripts/06_extract_phishing.py")

        df = pd.read_csv(path, low_memory=False)
        subject = df.get("subject", pd.Series("", index=df.index)).fillna("")
        body = df.get("body", pd.Series("", index=df.index)).fillna("")

        out = pd.DataFrame({
            "text": (subject.astype(str) + "\n" + body.astype(str)).str.strip(),
            "label": df["label"].astype(int),
            "source": name,
        })

        if has_date:
            # Email Date: headers are inconsistent; unparseable ones become NaT
            # rather than silently shifting a message to the wrong era.
            out["date"] = pd.to_datetime(df["date"], errors="coerce",
                                         utc=True, format="mixed")
        else:
            out["date"] = pd.NaT

        frames.append(out)

    full = pd.concat(frames, ignore_index=True)
    full = full[full["text"].str.len() > 0].reset_index(drop=True)
    return full


def corpus_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-corpus size, class balance and date span.

    Read the `pct_phish` column carefully: a corpus that is 0% or 100% phishing
    contributes only one class, so any model can separate it by style alone.
    """
    rows = []
    for name, sub in df.groupby("source"):
        dates = sub["date"].dropna()
        rows.append({
            "source": name,
            "n": len(sub),
            "pct_phish": round(100.0 * sub["label"].mean(), 1),
            "n_dated": len(dates),
            "first": dates.min().date() if len(dates) else None,
            "last": dates.max().date() if len(dates) else None,
            "med_chars": int(sub["text"].str.len().median()),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


VALID_LO = pd.Timestamp("1995-01-01", tz="UTC")
VALID_HI = pd.Timestamp("2025-01-01", tz="UTC")


def temporal_split(df: pd.DataFrame, test_size: float = 0.25):
    """Train on older mail, test on newer - the honest drift evaluation.

    Undated messages are dropped: assigning them to either side would invent a
    chronology the data does not have. Dates outside [1995, 2025) are dropped
    too - Date: headers are attacker-controlled, and this archive contains
    years as implausible as 0102 and 2100.

    WARNING: on this particular collection the result is not interpretable.
    Ham appears only in 2002 (SpamAssassin) and 2008 (CEAS); every other year
    is ~100% phishing, so any temporal cut yields a near-single-class test set
    that a constant "phishing" prediction scores well on. Date is confounded
    with label here. Use leave_one_corpus_out() instead, and see
    scripts/08_check_dates.py for the evidence.
    """
    dated = df[df["date"].notna()]
    dated = dated[dated["date"].between(VALID_LO, VALID_HI)]
    dated = dated.sort_values("date").reset_index(drop=True)
    if dated.empty:
        raise ValueError("no plausibly-dated messages; "
                         "use sources from corpora.DATED")

    cut = int(len(dated) * (1 - test_size))
    boundary = dated.loc[cut, "date"]
    train = dated[dated["date"] < boundary].reset_index(drop=True)
    test = dated[dated["date"] >= boundary].reset_index(drop=True)
    return train, test, boundary


def leave_one_corpus_out(df: pd.DataFrame, holdout: str):
    """Train on every corpus except `holdout`, test on it.

    This is the strongest available check that a model generalises across
    sources rather than memorising the quirks of one collection.
    """
    if holdout not in set(df["source"]):
        raise ValueError(f"unknown corpus: {holdout}")
    train = df[df["source"] != holdout].reset_index(drop=True)
    test = df[df["source"] == holdout].reset_index(drop=True)
    return train, test


def subsample_to_prior(df: pd.DataFrame, phish_rate: float = 0.05,
                       seed: int = SEED) -> pd.DataFrame:
    """Down-sample phishing to a realistic base rate.

    The corpus as shipped is ~52% phishing. Real mailflow is a few percent, and
    precision at a realistic prior is far lower than the balanced number
    suggests.
    """
    rng = np.random.default_rng(seed)
    ham = df[df["label"] == 0]
    phish = df[df["label"] == 1]
    target = int(len(ham) * phish_rate / (1 - phish_rate))
    if target < len(phish):
        keep = rng.choice(phish.index.values, size=target, replace=False)
        phish = phish.loc[keep]
    return (pd.concat([ham, phish])
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True))
