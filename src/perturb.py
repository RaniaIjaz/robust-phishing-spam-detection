"""Evasion-style text perturbations.

Spam and phishing authors actively rewrite messages to slip past filters while
keeping them readable to a human. Each function here reproduces one such
technique, so a classifier can be scored on adversarially rewritten text
rather than only on the clean corpus.

All functions are deterministic given `seed`.
"""
from __future__ import annotations

import random

# Cyrillic / Greek characters that render almost identically to Latin ones.
HOMOGLYPHS = {
    "a": "а", "c": "с", "e": "е", "i": "і",
    "j": "ј", "o": "о", "p": "р", "s": "ѕ",
    "x": "х", "y": "у",
}

ZERO_WIDTH = ["​", "‌", "‍", "﻿"]

LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "b": "8"}


def _targets(text: str, eligible: set[str], rate: float,
             rng: random.Random) -> set[int]:
    """Pick which character positions to modify."""
    idx = [i for i, ch in enumerate(text) if ch.lower() in eligible]
    if not idx:
        return set()
    k = max(1, int(round(len(idx) * rate)))
    return set(rng.sample(idx, min(k, len(idx))))


def homoglyph(text: str, rate: float = 0.15, seed: int = 0) -> str:
    """Swap Latin letters for visually identical non-Latin codepoints."""
    rng = random.Random(seed)
    hit = _targets(text, set(HOMOGLYPHS), rate, rng)
    return "".join(HOMOGLYPHS[ch.lower()] if i in hit else ch
                   for i, ch in enumerate(text))


def zero_width(text: str, rate: float = 0.15, seed: int = 0) -> str:
    """Insert invisible characters, breaking tokens without visible change."""
    rng = random.Random(seed)
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < rate:
            out.append(rng.choice(ZERO_WIDTH))
    return "".join(out)


def leetspeak(text: str, rate: float = 0.20, seed: int = 0) -> str:
    """Replace letters with lookalike digits (fr33 m0n3y)."""
    rng = random.Random(seed)
    hit = _targets(text, set(LEET), rate, rng)
    return "".join(LEET[ch.lower()] if i in hit else ch
                   for i, ch in enumerate(text))


def typo(text: str, rate: float = 0.08, seed: int = 0) -> str:
    """Transpose adjacent characters inside words."""
    rng = random.Random(seed)
    chars = list(text)
    positions = [i for i in range(len(chars) - 1)
                 if chars[i].isalpha() and chars[i + 1].isalpha()]
    if not positions:
        return text
    k = max(1, int(round(len(positions) * rate)))
    for i in sorted(rng.sample(positions, min(k, len(positions))),
                    reverse=True):
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def spacing(text: str, rate: float = 0.15, seed: int = 0) -> str:
    """Split words with spaces or dots (F R E E, c.l.i.c.k)."""
    rng = random.Random(seed)
    words = text.split()
    if not words:
        return text
    k = max(1, int(round(len(words) * rate)))
    for i in rng.sample(range(len(words)), min(k, len(words))):
        if len(words[i]) > 3:
            sep = rng.choice([" ", "."])
            words[i] = sep.join(words[i])
    return " ".join(words)


ATTACKS = {
    "homoglyph": homoglyph,
    "zero_width": zero_width,
    "leetspeak": leetspeak,
    "typo": typo,
    "spacing": spacing,
}


def apply_attack(texts, name: str, rate: float, seed: int = 0) -> list[str]:
    """Apply one named attack to every message, varying the per-message seed."""
    fn = ATTACKS[name]
    return [fn(t, rate=rate, seed=seed + i) for i, t in enumerate(texts)]
