"""Dataset loading and train/test splitting.

Two splitting strategies are provided so the experiments can quantify how much
of a reported score is real generalisation and how much is leakage:

  random_split  - the conventional i.i.d. shuffle used by most tutorials.
  grouped_split - near-duplicate messages are clustered first, and whole
                  clusters are assigned to one side of the split, so no
                  template seen in training reappears in test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DUPLICATE_THRESHOLD, SEED, TEST_SIZE


def load_sms_spam() -> pd.DataFrame:
    """Load the UCI SMS Spam Collection as a DataFrame[text, label].

    label is 1 for spam, 0 for ham.
    """
    path = DATA_RAW / "SMSSpamCollection"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/01_download_data.py"
        )
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                     encoding="utf-8", quoting=3)
    df["label"] = (df["label"] == "spam").astype(int)
    df["text"] = df["text"].astype(str)
    return df.reset_index(drop=True)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def normalised_key(text: str) -> str:
    """Aggressive normalisation for cheap duplicate detection.

    Lowercases, strips digits and punctuation, collapses whitespace. Two
    template sends that differ only in a name, an amount or a tracking code
    collapse to the same key.
    """
    out = [c if c.isalpha() or c.isspace() else " " for c in text.lower()]
    return " ".join("".join(out).split())


def _hash_clusters(texts: list[str]) -> np.ndarray:
    """O(n) duplicate clustering by normalised-text equality."""
    seen: dict[str, int] = {}
    ids = np.empty(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        ids[i] = seen.setdefault(normalised_key(t), len(seen))
    return ids


def duplicate_clusters(texts: list[str],
                       threshold: float = DUPLICATE_THRESHOLD,
                       exact_above: int = 20_000) -> np.ndarray:
    """Assign each message a cluster id, merging near-duplicates.

    Similarity is cosine on L2-normalised character 5-gram TF-IDF, which is
    robust to the small edits (names, amounts, short links) that spam
    templates vary between sends.

    That comparison is O(n^2). Above `exact_above` messages it becomes the
    dominant cost of the whole pipeline, so the function falls back to
    normalised-text hashing, which is linear but only catches duplicates that
    survive normalisation. The fallback under-counts; it never invents
    clusters, so a grouped split built on it stays conservative.
    """
    if len(texts) > exact_above:
        return _hash_clusters(texts)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(5, 5),
                          min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)

    uf = _UnionFind(len(texts))
    # Blocked matrix product keeps peak memory bounded on larger corpora.
    block = 512
    for start in range(0, X.shape[0], block):
        sim = (X[start:start + block] @ X.T).toarray()
        rows, cols = np.nonzero(sim >= threshold)
        for r, c in zip(rows, cols):
            i = start + r
            if i < c:  # upper triangle only; avoids self-pairs and repeats
                uf.union(i, int(c))

    roots = np.array([uf.find(i) for i in range(len(texts))])
    # Relabel to contiguous ids for readability.
    _, cluster_ids = np.unique(roots, return_inverse=True)
    return cluster_ids


def random_split(df: pd.DataFrame, test_size: float = TEST_SIZE,
                 seed: int = SEED):
    """Conventional stratified i.i.d. split."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed,
                                   stratify=df["label"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def grouped_split(df: pd.DataFrame, test_size: float = TEST_SIZE,
                  seed: int = SEED):
    """Leakage-aware split: a duplicate cluster never straddles the boundary.

    Clusters are shuffled and greedily assigned to test until the target size
    is reached, so the test set contains only templates absent from training.
    """
    clusters = duplicate_clusters(df["text"].tolist())
    df = df.assign(cluster=clusters)

    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(clusters))

    sizes = df.groupby("cluster").size()
    target = int(len(df) * test_size)

    test_clusters, running = set(), 0
    for c in unique:
        if running >= target:
            break
        test_clusters.add(int(c))
        running += int(sizes[c])

    mask = df["cluster"].isin(test_clusters)
    return (df[~mask].reset_index(drop=True),
            df[mask].reset_index(drop=True))


def leakage_report(df: pd.DataFrame) -> dict:
    """Summarise how much near-duplication the corpus actually contains."""
    clusters = duplicate_clusters(df["text"].tolist())
    sizes = pd.Series(clusters).value_counts()
    return {
        "n_messages": int(len(df)),
        "n_clusters": int(sizes.size),
        "n_in_multi_clusters": int(sizes[sizes > 1].sum()),
        "pct_duplicated": round(100.0 * sizes[sizes > 1].sum() / len(df), 2),
        "largest_cluster": int(sizes.iloc[0]),
    }
