"""Defences against the evasion attacks in perturb.py.

Experiment 2 showed what breaks; this module tries to fix it. Two families of
defence, deliberately different in kind:

  normalise        - undo the obfuscation *before* the classifier sees the
                     text. Rule-based, cheap, explainable, and only as good as
                     its rules: an attacker who uses a trick the rules do not
                     know walks straight past it.

  adversarial      - leave the input alone and teach the classifier what
  training           obfuscated spam looks like, by adding perturbed copies
                     of the training messages. No rules to step around, but
                     it can only learn tricks it has been shown.

Both are wired in as ordinary scikit-learn pipeline steps or data
transformations, so a defended model is scored exactly like an undefended one.
"""
from __future__ import annotations

import random
import re
import unicodedata

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SEED
from src.perturb import ATTACKS

# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

# Format characters that render as nothing. Zero-width space/joiners, the BOM,
# word joiner and soft hyphen are the ones seen in real evasion.
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))

# Common Cyrillic and Greek letters that render like Latin ones. A defender
# would build this from a public confusables list; it is not derived from the
# attack's own table, although the common Cyrillic cases necessarily overlap.
# The held-out homoglyph attack in perturb.py uses characters outside it.
_CONFUSABLES = {
    # Cyrillic lower
    "а": "a", "в": "b", "с": "c", "е": "e", "һ": "h", "і": "i", "ј": "j",
    "к": "k", "м": "m", "н": "h", "о": "o", "р": "p", "ԛ": "q", "ѕ": "s",
    "т": "t", "ԝ": "w", "х": "x", "у": "y", "ԁ": "d",
    # Cyrillic upper
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "Ј": "J",
    "К": "K", "М": "M", "О": "O", "Р": "P", "Ѕ": "S", "Т": "T", "Х": "X",
    "У": "Y",
    # Greek lower
    "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p",
    "τ": "t", "υ": "u", "χ": "x",
    # Greek upper
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
_CONFUSABLES_TABLE = str.maketrans(_CONFUSABLES)

_DELEET = {"4": "a", "3": "e", "1": "i", "0": "o",
           "5": "s", "7": "t", "8": "b"}

# Three or more single characters joined by dots ("c.l.i.c.k") or by single
# spaces ("F R E E"). Requiring three avoids "U.K", "e.g." and "V r".
# Dotted runs are rejoined first because their word boundaries are
# unambiguous; a run of spaced letters next to another one cannot be split.
_DOTTED = re.compile(r"(?<![\w.])(?:[^\W_]\.){2,}[^\W_](?!\w)")
_SPACED = re.compile(r"(?<!\S)(?:[^\W_] ){2,}[^\W_](?!\S)")
_ALNUM_RUN = re.compile(r"[^\W_]+")
_LETTER = r"[^\W\d_]"
_LEET_INSIDE = re.compile(rf"(?<={_LETTER})[0134578]+(?={_LETTER})")
_LEET_END = re.compile(rf"(?<={_LETTER}{_LETTER})[0134578]+$")


def _deleet_run(run: str) -> str:
    """Map digits back to letters only where a letter is expected.

    SMS spam is full of real numbers ("150p", "08452810075", "4giv", "1hr")
    that are themselves strong spam and ham signals; rewriting those would
    hurt more than the leetspeak attack does. A digit is rewritten only when
    it sits between two letters ("fr3e"), or ends a word after at least two
    letters ("attemp7"). Leading digits are never touched. Consecutive
    leet digits ("fr33") are treated as one block.
    """
    def sub(m):
        return "".join(_DELEET[c] for c in m.group(0))
    run = _LEET_INSIDE.sub(sub, run)
    return _LEET_END.sub(sub, run)


def normalise(text: str) -> str:
    """Undo the obfuscation tricks that can be undone by rule.

    Order matters: invisible characters go first so they cannot hide a
    homoglyph or split a spaced-out word; NFKC folds full-width and
    mathematical-alphabet letters to plain Latin before the confusables map.

    Character transposition (`typo`) has no rule-based inverse and is left
    alone; only adversarial training addresses it.
    """
    text = text.translate(_INVISIBLE)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CONFUSABLES_TABLE)
    text = _DOTTED.sub(lambda m: m.group(0).replace(".", ""), text)
    text = _SPACED.sub(lambda m: m.group(0).replace(" ", ""), text)
    return _ALNUM_RUN.sub(lambda m: _deleet_run(m.group(0)), text)


def normalise_batch(texts) -> list[str]:
    return [normalise(t) for t in texts]


def with_normaliser(model: Pipeline) -> Pipeline:
    """Wrap a pipeline so every input is normalised at fit *and* predict time.

    Normalising only at inference would hand the classifier text from a
    different distribution than it was trained on.
    """
    return Pipeline([
        ("normalise", FunctionTransformer(normalise_batch, validate=False)),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# Adversarial training
# ---------------------------------------------------------------------------

TRAIN_RATES = [0.05, 0.10, 0.20, 0.35, 0.50]


def augment(texts, labels, attacks, seed: int = SEED + 1000):
    """Return the training set plus one perturbed copy per message per attack.

    Each copy gets its own random rate from TRAIN_RATES, so the model sees the
    whole range from light to heavy obfuscation. Ham is perturbed as well as
    spam: if only spam were obfuscated, the model could learn the shortcut
    "unusual characters mean spam", which is exactly the false-positive
    failure the email model showed in its demo.

    The default seed differs from the one used to attack the test set, so the
    model is never trained on the exact perturbations it is scored on.
    """
    texts, labels = list(texts), list(labels)
    rng = random.Random(seed)
    out_t, out_y = list(texts), list(labels)
    for name in attacks:
        fn = ATTACKS[name]
        for i, (t, y) in enumerate(zip(texts, labels)):
            out_t.append(fn(t, rate=rng.choice(TRAIN_RATES),
                            seed=seed + 7919 * (i + 1)))
            out_y.append(y)
    return out_t, out_y
