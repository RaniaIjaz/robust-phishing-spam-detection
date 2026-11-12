"""End-to-end smoke test: every module imports, every model fits and scores.

Exercises both duplicate-clustering paths and all five attacks. Fast enough to
run after any edit.

Usage:  python scripts/10_smoke_test.py
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data import (duplicate_clusters, grouped_split, leakage_report,
                      load_sms_spam, normalised_key, random_split)
from src.evaluate import best_threshold, metrics_at, scores, top_errors
from src.models import MODELS, build_model
from src.perturb import ATTACKS, apply_attack

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  ok   {name}")
    except Exception:
        failures.append(name)
        print(f"  FAIL {name}")
        traceback.print_exc()


def main() -> int:
    print("loading SMS corpus")
    df = load_sms_spam()
    small = df.sample(n=600, random_state=0).reset_index(drop=True)

    print("\nsplitting")
    check("random_split", lambda: random_split(small))
    check("grouped_split", lambda: grouped_split(small))
    check("leakage_report", lambda: leakage_report(small))

    print("\nduplicate clustering, both paths")

    def exact_path():
        ids = duplicate_clusters(small["text"].tolist(), exact_above=10_000)
        assert len(ids) == len(small)

    def hash_path():
        # exact_above below n forces the linear fallback
        ids = duplicate_clusters(small["text"].tolist(), exact_above=10)
        assert len(ids) == len(small)

    check("duplicate_clusters exact", exact_path)
    check("duplicate_clusters hash fallback", hash_path)
    check("normalised_key", lambda: normalised_key("Call 0800-123 NOW!!"))

    print("\nattacks")
    for attack in ATTACKS:
        def run(a=attack):
            out = apply_attack(small["text"].tolist()[:20], a, 0.3, seed=0)
            assert len(out) == 20
            assert any(o != t for o, t in zip(out, small["text"][:20]))
        check(f"attack {attack}", run)

    print("\nmodels: fit, score, threshold, metrics")
    train, test = random_split(small)
    for name in MODELS + ["logreg_email"]:
        def run(n=name):
            m = build_model(n)
            m.fit(train["text"], train["label"])
            s = scores(m, test["text"])
            assert s.shape == (len(test),)
            assert np.all((s >= 0) & (s <= 1))
            thr, _ = best_threshold(train["label"], scores(m, train["text"]))
            met = metrics_at(test["label"], s, thr)
            assert 0.0 <= met["f1"] <= 1.0
        check(f"model {name}", run)

    print("\nerror dump")
    m = build_model("logreg_word")
    m.fit(train["text"], train["label"])
    s = scores(m, test["text"])
    check("top_errors fn",
          lambda: top_errors(test["text"], test["label"], s, 0.5, "fn"))
    check("top_errors fp",
          lambda: top_errors(test["text"], test["label"], s, 0.5, "fp"))

    print("\ndefences")
    from src.defences import augment, normalise, with_normaliser
    from src.perturb import HELD_OUT_ATTACKS

    def held_out_attack():
        from src.perturb import UNSEEN_HOMOGLYPHS
        out = apply_attack(["please call now"], "homoglyph_unseen", 1.0)
        assert out[0] != "please call now"
        # NFKC may fold some of them (lunate sigma -> final sigma), but never
        # back to the Latin letter they imitate.
        for latin, glyph in UNSEEN_HOMOGLYPHS.items():
            assert normalise(glyph) != latin, (glyph, latin)
    check("held-out homoglyph attack", held_out_attack)

    def normalise_undoes_attacks():
        s = "free prize call now"
        for a in ["zero_width", "homoglyph"]:
            got = normalise(apply_attack([s], a, 1.0, seed=1)[0])
            assert got == s, (a, got)
        # Two adjacent spaced-out words cannot be told apart ("f r e e
        # p r i z e"), so spacing is checked on one attackable word.
        for seed in range(4):
            got = normalise(apply_attack(["prize now"], "spacing", 1.0,
                                         seed=seed)[0])
            assert got == "prize now", got
        assert normalise("fr33 pr1z3") == "free prize"
    check("normalise undoes zero-width, homoglyph, spacing, leet",
          normalise_undoes_attacks)

    def normalise_keeps_numbers():
        for s in ["call 09061701461", "150p/msg", "1hr", "4giv", "U.K"]:
            assert normalise(s) == s, (s, normalise(s))
    check("normalise leaves real numbers alone", normalise_keeps_numbers)

    def augment_sizes():
        t, y = augment(["a b c", "d e f"], [0, 1], ["typo", "spacing"])
        assert len(t) == len(y) == 6 and y == [0, 1, 0, 1, 0, 1]
    check("augment adds one copy per message per attack", augment_sizes)

    def normalised_model():
        m = with_normaliser(build_model("logreg_char"))
        m.fit(train["text"], train["label"])
        assert scores(m, test["text"]).shape == (len(test),)
    check("with_normaliser fits and scores", normalised_model)
    assert set(HELD_OUT_ATTACKS).isdisjoint(ATTACKS)

    print("\nword embeddings (glove excluded: it needs a ~130 MB download)")
    from src.embeddings import build_embedding_model, char_ngrams, tokenize

    def zero_width_splits():
        got = tokenize("fr\u200bee")
        assert got == ["fr", "ee"], got
    check("tokenize splits on zero-width", zero_width_splits)

    def ngrams():
        got = char_ngrams("free", 3, 3)
        assert got == ["<fr", "fre", "ree", "ee>"], got
    check("char_ngrams", ngrams)

    fitted_ft = {}
    for source in ["word2vec", "fasttext"]:
        def run(s=source):
            m = build_embedding_model(s)
            m.fit(train["text"], train["label"])
            p = scores(m, test["text"])
            assert p.shape == (len(test),)
            assert np.all((p >= 0) & (p <= 1))
            fitted_ft[s] = m
        check(f"embedding {source}", run)

    def fasttext_covers_oov():
        embed = fitted_ft["fasttext"].named_steps["embed"]
        vec = embed.transform(["zzqqxxunseenword"])
        assert np.any(vec != 0), "fasttext gave no vector for an OOV word"
    check("fasttext vectors for unseen words", fasttext_covers_oov)

    def word2vec_drops_oov():
        embed = fitted_ft["word2vec"].named_steps["embed"]
        vec = embed.transform(["zzqqxxunseenword"])
        assert np.all(vec == 0), "word2vec should have no vector for OOV"
    check("word2vec has no vector for unseen words", word2vec_drops_oov)

    print("\ncorpora module (skipped if archive not extracted)")
    try:
        from src.corpora import (PHISHING_DIR, corpus_profile,
                                 leave_one_corpus_out, load_phishing,
                                 subsample_to_prior, temporal_split)
        if not PHISHING_DIR.exists():
            print("  skip  phishing archive not extracted")
        else:
            e = load_phishing(sources=["nazario", "spamassassin"])
            check("load_phishing", lambda: e)
            check("corpus_profile", lambda: corpus_profile(e))
            check("leave_one_corpus_out",
                  lambda: leave_one_corpus_out(e, "nazario"))
            check("temporal_split", lambda: temporal_split(e))
            check("subsample_to_prior", lambda: subsample_to_prior(e, 0.05))
    except Exception:
        failures.append("corpora import")
        traceback.print_exc()

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
