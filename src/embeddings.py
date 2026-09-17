"""Word embeddings, implemented from scratch in NumPy.

gensim has no build for Python 3.14 and this machine has no C compiler, so the
two trained models are written directly:

  word2vec  skip-gram with negative sampling (Mikolov et al., 2013)
  fasttext  the same objective, but a word's input vector is the average of
            vectors for the word itself and its character 3-5-grams
            (Bojanowski et al., 2017)
  glove     pretrained GloVe, 100d, Wikipedia + Gigaword
            (Pennington et al., 2014), downloaded once

Each message becomes the mean of its word vectors, and a logistic regression
classifies that vector.

The question tested in scripts/13_embeddings.py is whole-word vs subword under
the attacks in perturb.py. GloVe and word2vec hold a vector only for tokens
seen in training; an attacked token such as "frее" (Cyrillic е) is out of
vocabulary and is dropped from the average. FastText composes a vector for any
string from its character n-grams, so the attacked token still gets one.

The expectation was that FastText would therefore behave like character
TF-IDF. It mostly did not: it only clearly beat word2vec on zero-width
evasion. Its guessed vectors for mangled words are noisy, and averaging noise
in can hurt more than dropping the word. See the README, "Fourth result".
"""
from __future__ import annotations

import csv
import gzip
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED, DATA_RAW, SEED

# Same rule as scikit-learn's default TF-IDF token_pattern (2+ word
# characters), so the TF-IDF vs embedding comparison differs only in the
# representation. An earlier version kept single characters; attacks that
# shatter words into letters then fed those letters into the average, and the
# embeddings looked less robust than TF-IDF for a reason that had nothing to
# do with embeddings.
TOKEN = re.compile(r"\b\w\w+\b")
SOURCES = ["glove", "word2vec", "fasttext"]

GLOVE_URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/"
             "download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz")
GLOVE_GZ = DATA_RAW / "glove-wiki-gigaword-100.gz"
GLOVE_NPY = DATA_PROCESSED / "glove100_vectors.npy"
GLOVE_VOCAB = DATA_PROCESSED / "glove100_vocab.txt"

_glove_cache = None


# --------------------------------------------------------------------------
# Tokens and subwords

def tokenize(text: str) -> list[str]:
    """Lowercase tokens of two or more word characters. Zero-width characters
    are not \\w, so they split a word into fragments - exactly the damage the
    attack intends."""
    return TOKEN.findall(text.lower())


def char_ngrams(word: str, min_n: int = 3, max_n: int = 5) -> list[str]:
    """FastText-style n-grams, with < and > marking the word boundaries."""
    w = f"<{word}>"
    return [w[i:i + n] for n in range(min_n, max_n + 1)
            for i in range(len(w) - n + 1)]


def _bucket(gram: str, buckets: int) -> int:
    """Hash an n-gram into a fixed number of rows, as FastText does."""
    return zlib.crc32(gram.encode("utf-8")) % buckets


# --------------------------------------------------------------------------
# Vector lookup

class Vectors:
    """Word -> vector table, with a subword fallback for unseen words."""

    def __init__(self, words: list[str], vectors: np.ndarray,
                 ngram_table: np.ndarray | None = None, buckets: int = 0,
                 min_n: int = 3, max_n: int = 5):
        self.key_to_index = {w: i for i, w in enumerate(words)}
        self.vectors = vectors
        self.vector_size = vectors.shape[1]
        self.ngram_table = ngram_table
        self.buckets = buckets
        self.min_n, self.max_n = min_n, max_n
        self._oov: dict[str, np.ndarray] = {}

    def __getitem__(self, word: str) -> np.ndarray:
        i = self.key_to_index.get(word)
        if i is not None:
            return self.vectors[i]
        if self.ngram_table is None:
            raise KeyError(word)
        if word not in self._oov:
            ids = [_bucket(g, self.buckets)
                   for g in char_ngrams(word, self.min_n, self.max_n)]
            self._oov[word] = self.ngram_table[ids].mean(axis=0)
        return self._oov[word]

    def most_similar(self, word: str, topn: int = 8) -> list[tuple[str, float]]:
        """Nearest vocabulary words by cosine similarity."""
        try:
            v = self[word]
        except KeyError:
            return []
        norms = np.linalg.norm(self.vectors, axis=1) + 1e-9
        sims = (self.vectors @ v) / (norms * (np.linalg.norm(v) + 1e-9))
        words = list(self.key_to_index)
        out = [(words[i], float(sims[i])) for i in np.argsort(-sims)
               if words[i] != word]
        return out[:topn]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / ((np.linalg.norm(a) + 1e-9) *
                          (np.linalg.norm(b) + 1e-9)))


# --------------------------------------------------------------------------
# Skip-gram with negative sampling (word2vec), optionally with subwords

def train_sgns(docs: list[list[str]], dim: int = 100, window: int = 5,
               negative: int = 5, epochs: int = 20, min_count: int = 2,
               lr: float = 0.025, sample: float = 1e-3,
               subword: bool = False, buckets: int = 50_000,
               min_n: int = 3, max_n: int = 5, batch: int = 1024,
               seed: int = SEED) -> Vectors:
    """Train skip-gram embeddings with negative sampling.

    For every (centre, context) pair within `window` tokens, push the centre's
    input vector towards the context's output vector, and away from the output
    vectors of `negative` words drawn from the unigram^0.75 distribution.

    With subword=True the centre's input vector is the mean of rows for the
    word and each of its character n-grams, and the gradient is shared back
    across all of them. That is the only difference between word2vec and
    FastText, and it is what lets FastText build vectors for unseen words.
    """
    rng = np.random.default_rng(seed)

    counts = Counter(w for d in docs for w in d)
    words = sorted((w for w, c in counts.items() if c >= min_count),
                   key=lambda w: (-counts[w], w))
    if len(words) < 2:
        raise ValueError("vocabulary too small to train embeddings")
    index = {w: i for i, w in enumerate(words)}
    V = len(words)

    seqs = [np.array([index[w] for w in d if w in index], dtype=np.int64)
            for d in docs]
    seqs = [s for s in seqs if len(s) > 1]

    freq = np.array([counts[w] for w in words], dtype=np.float64)
    f = freq / freq.sum()
    # Frequent-word subsampling: "the" and "you" are discarded most of the time.
    keep_prob = np.minimum(1.0, (np.sqrt(f / sample) + 1) * sample / f)
    noise_cdf = np.cumsum(freq ** 0.75)
    noise_cdf /= noise_cdf[-1]

    if subword:
        rows = [[i] + [V + _bucket(g, buckets)
                       for g in char_ngrams(w, min_n, max_n)]
                for i, w in enumerate(words)]
        width = max(len(r) for r in rows)
        ids = np.zeros((V, width), dtype=np.int64)
        mask = np.zeros((V, width), dtype=bool)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = r
            mask[i, :len(r)] = True
        n_ids = mask.sum(axis=1).astype(np.float32)
        n_in = V + buckets
    else:
        n_in = V

    W_in = rng.uniform(-0.5 / dim, 0.5 / dim,
                       size=(n_in, dim)).astype(np.float32)
    W_out = np.zeros((V, dim), dtype=np.float32)

    def epoch_pairs():
        centres, contexts = [], []
        for s in seqs:
            s = s[rng.random(len(s)) < keep_prob[s]]
            for k in range(1, min(window, len(s) - 1) + 1):
                centres += [s[:-k], s[k:]]
                contexts += [s[k:], s[:-k]]
        if not centres:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        c, o = np.concatenate(centres), np.concatenate(contexts)
        perm = rng.permutation(len(c))
        return c[perm], o[perm]

    step, total = 0, None
    for _ in range(epochs):
        c_all, o_all = epoch_pairs()
        if total is None:
            total = max(1, epochs * -(-len(c_all) // batch))
        for start in range(0, len(c_all), batch):
            c = c_all[start:start + batch]
            o = o_all[start:start + batch]
            alpha = np.float32(lr * max(1e-4, 1.0 - step / total))
            step += 1

            neg = np.minimum(np.searchsorted(noise_cdf,
                                             rng.random((len(c), negative))),
                             V - 1)
            targets = np.concatenate([o[:, None], neg], axis=1)  # (B, 1+K)

            if subword:
                rid, m, cnt = ids[c], mask[c], n_ids[c]
                h = (W_in[rid] * m[..., None]).sum(axis=1) / cnt[:, None]
            else:
                h = W_in[c]

            U = W_out[targets]                                    # (B, 1+K, d)
            logits = np.clip(np.einsum("bkd,bd->bk", U, h), -6.0, 6.0)
            label = np.zeros_like(logits)
            label[:, 0] = 1.0
            g = (label - 1.0 / (1.0 + np.exp(-logits))) * alpha   # (B, 1+K)

            grad_h = np.einsum("bk,bkd->bd", g, U)
            np.add.at(W_out, targets.ravel(),
                      (g[..., None] * h[:, None, :]).reshape(-1, dim))
            if subword:
                share = np.repeat(grad_h / cnt[:, None], m.sum(axis=1), axis=0)
                np.add.at(W_in, rid[m], share)
            else:
                np.add.at(W_in, c, grad_h)

    if not subword:
        return Vectors(words, W_in)

    word_vecs = np.empty((V, dim), dtype=np.float32)
    for s in range(0, V, 2048):
        r, m = ids[s:s + 2048], mask[s:s + 2048]
        word_vecs[s:s + 2048] = ((W_in[r] * m[..., None]).sum(axis=1)
                                 / m.sum(axis=1, keepdims=True))
    return Vectors(words, word_vecs, ngram_table=W_in[V:], buckets=buckets,
                   min_n=min_n, max_n=max_n)


# --------------------------------------------------------------------------
# Pretrained GloVe

def _download(url: str, dest: Path) -> None:
    import requests

    print(f"  downloading GloVe (~130 MB) to {dest}")
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)


def load_glove() -> Vectors:
    """GloVe 100d. Downloaded and parsed once, then cached as .npy."""
    global _glove_cache
    if _glove_cache is not None:
        return _glove_cache

    if not GLOVE_NPY.exists():
        if not GLOVE_GZ.exists():
            _download(GLOVE_URL, GLOVE_GZ)
        print("  parsing GloVe (one-off)")
        with gzip.open(GLOVE_GZ, "rt", encoding="utf-8") as fh:
            first = fh.readline().split()
        has_header = len(first) == 2 and all(t.isdigit() for t in first)
        dim = int(first[1]) if has_header else len(first) - 1
        table = pd.read_csv(
            GLOVE_GZ, sep=" ", header=None, skiprows=1 if has_header else 0,
            quoting=csv.QUOTE_NONE, keep_default_na=False, na_filter=False,
            encoding="utf-8", usecols=range(dim + 1),
            dtype={0: str, **{i: np.float32 for i in range(1, dim + 1)}})
        np.save(GLOVE_NPY, table.iloc[:, 1:].to_numpy(np.float32))
        GLOVE_VOCAB.write_text("\n".join(table[0]), encoding="utf-8")

    vectors = np.load(GLOVE_NPY)
    words = GLOVE_VOCAB.read_text(encoding="utf-8").split("\n")
    if len(words) != len(vectors):
        raise ValueError(f"GloVe cache mismatch: {len(words)} words, "
                         f"{len(vectors)} vectors; delete {GLOVE_NPY}")
    _glove_cache = Vectors(words, vectors)
    return _glove_cache


# --------------------------------------------------------------------------
# scikit-learn interface

class MeanEmbedding(BaseEstimator, TransformerMixin):
    """Average the word vectors of each message into one dense vector."""

    def __init__(self, source: str = "fasttext", dim: int = 100,
                 epochs: int = 20, min_count: int = 2, seed: int = SEED):
        self.source = source
        self.dim = dim
        self.epochs = epochs
        self.min_count = min_count
        self.seed = seed

    def fit(self, X, y=None):
        if self.source == "glove":
            self.kv_ = load_glove()
        elif self.source in ("word2vec", "fasttext"):
            self.kv_ = train_sgns([tokenize(t) for t in X], dim=self.dim,
                                  epochs=self.epochs,
                                  min_count=self.min_count,
                                  subword=self.source == "fasttext",
                                  seed=self.seed)
        else:
            raise ValueError(f"unknown embedding source: {self.source}")
        return self

    def transform(self, X):
        kv = self.kv_
        index = kv.key_to_index
        out = np.zeros((len(X), kv.vector_size), dtype=np.float32)
        for i, text in enumerate(X):
            tokens = tokenize(text)
            if not tokens:
                continue
            if self.source == "fasttext":
                # Subword model: every token gets a vector, seen or not.
                out[i] = np.mean([kv[w] for w in tokens], axis=0)
            else:
                # Whole-word models: unknown tokens are silently dropped.
                idx = [index[w] for w in tokens if w in index]
                if idx:
                    out[i] = kv.vectors[idx].mean(axis=0)
        return out

    def oov_rate(self, X) -> float:
        """Share of tokens with no whole-word vector.

        Reported for FastText too: its subword vectors still cover these
        tokens, which is the point of the comparison.
        """
        total = missing = 0
        for text in X:
            for w in tokenize(text):
                total += 1
                missing += w not in self.kv_.key_to_index
        return missing / total if total else 0.0


def build_embedding_model(source: str) -> Pipeline:
    return Pipeline([
        ("embed", MeanEmbedding(source=source)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=SEED)),
    ])
