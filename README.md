# Robustness-Aware Text Classification for Phishing and Spam Detection

Building a spam classifier is easy. Building one whose reported accuracy means
anything is not. This project builds classifiers on two corpora and then
attacks its own results three ways:

1. **Duplicate leakage** - are near-identical messages leaking across the
   train/test boundary?
2. **Corpus leakage** - when a benchmark is assembled from several sources, is
   the model detecting phishing or detecting *which corpus a message is from*?
3. **Adversarial evasion** - does it survive an attacker who rewrites messages
   specifically to slip past it?

All three answers are no. The third is the one that would matter in
deployment; the second is the one that most published results get wrong.

It then compares TF-IDF against word embeddings (GloVe, plus word2vec and
FastText implemented from scratch) on the same three questions, and finally
builds defences against the evasion attacks and tests whether those defences
hold up against attacks they were not built for (mostly, they do not).

---

## Headline result: a 0.987 benchmark that misses 1 in 3 real phishing emails

The Kaggle phishing collection (82,486 emails) pools six source corpora. Four
evaluations of the same data, in increasing order of honesty:

| | Evaluation | Score |
|---|---|---|
| **A** | Corpus-identity probe - predict *which corpus* from text alone | **0.919** accuracy (chance = 0.167) |
| **B** | Naive pooled random split | **F1 0.9873** |
| **C** | Leave-one-corpus-out | **F1 0.83 - 0.93** |
| **D** | Within a single corpus | **F1 0.97 - 0.9965** |

**A explains everything else.** A linear model identifies the source corpus
with 92% accuracy against a 16.7% baseline. The six collections differ
enormously in era, encoding and house style — so a classifier handed the
pooled benchmark can score 0.987 by learning *corpus fingerprints* that
correlate with the label, never learning what phishing looks like.

That shortcut is invisible in B and D, because their test sets contain the
same corpora as training. It only appears in C:

| Held-out corpus | Recall | F1 | |
|---|---|---|---|
| `nigerian` | **0.9910** | — | single-class: recall only |
| `spamassassin` | 0.9598 | 0.8774 | |
| `ceas` | 0.9435 | 0.9290 | |
| `ling` | 0.8472 | 0.8472 | |
| `enron` | 0.8274 | 0.8305 | |
| `nazario` | **0.6575** | — | single-class: recall only |

**The `nazario` row is the finding.** That corpus is real captured phishing
mail from 2015-2022 — the most modern and most operationally realistic data in
the archive. A model trained on the other five corpora **misses 34% of it.**
The same model advertises F1 0.9873 on the pooled benchmark.

The contrast with `nigerian` (0.991 recall) is instructive. Advance-fee fraud
is formulaic and lexically distinctive, so it transfers almost perfectly.
Nazario's credential-harvesting phishing deliberately *imitates* legitimate
business email — it sits close to the ham it is hiding among, which is exactly
why it is both harder and more important to catch.

---

## Second result: word features collapse under evasion, character features do not

On the SMS corpus, trained on clean text and tested on adversarially rewritten
text (spam recall at 50% perturbation):

| Attack | `logreg_word` | `logreg_char` |
|---|---|---|
| zero-width characters | 0.972 → **0.644** | 0.966 → **0.966** |
| character transposition | 0.972 → **0.763** | 0.966 → **0.944** |
| homoglyph substitution | 0.972 → **0.870** | 0.966 → **0.961** |
| leetspeak | 0.972 → **0.864** | 0.966 → **0.977** |

Invisible characters split most words into fragments the word model has never
seen. Character n-grams that contain an inserted character stop matching too,
but the many short 2-3-character n-grams between insertions still match, so
most of the signal survives. Character n-grams are not a
marginal accuracy tweak — they are the difference between a filter that works
under attack and one that does not.

Curiosity worth noting: `logreg_char` *improves* under leetspeak
(0.966 → 0.977). Spam already contains digits, so leetspeak pushes messages
further from ham. Reported as an artifact rather than hidden.

Full curves: `results/robustness.png`.

---

## Third result: near-duplicate leakage reorders the model ranking

On the SMS corpus, 18.3% of messages sit in near-duplicate clusters (one
template appears 31 times). Splitting randomly puts copies of the same message
on both sides of the boundary:

| Model | Random split F1 | Leakage-aware F1 | Δ |
|---|---|---|---|
| `svm_union` | **0.9783** | 0.9218 | **−0.057** |
| `logreg_union` | 0.9677 | **0.9577** | −0.010 |
| `logreg_char` | 0.9598 | 0.9344 | −0.025 |
| `logreg_word` | 0.9355 | 0.8935 | −0.042 |
| `nb_word` | 0.8973 | 0.8602 | −0.037 |

The model that wins the naive evaluation is **fourth** under the honest one.

---

## Fourth result: word embeddings lose to TF-IDF, and my robustness prediction was wrong

Three embedding sources. Each message becomes the mean of its word vectors,
and a logistic regression classifies that vector:

- **GloVe** 100d, pretrained on Wikipedia + Gigaword; never sees this data
- **word2vec**, skip-gram with negative sampling, trained on the SMS training split
- **FastText**, the same objective plus character 3-5-gram subwords

gensim has no build for Python 3.14 and this machine has no C compiler, so
word2vec and FastText are implemented from scratch in NumPy
(`src/embeddings.py`). Nearest neighbours confirm they learned something:

| Word | GloVe (general English) | word2vec (SMS) | FastText (SMS) |
|---|---|---|---|
| `free` | through, without, allowed | replying, activate, tariffs | tone, chat, ringtone |
| `call` | calls, calling, ask | 11, 09056242159, landline | caller, calls, callers |
| `prize` | nobel, prizes, award | 2000, 1450, 09066358152 | 2000, awarded, 1000 |
| `love` | me, passion, my | kisses, mis, compromised | wish, loverboy, hear |

### Clean performance (SMS, leakage-aware split)

| Representation | F1 | PR-AUC |
|---|---|---|
| TF-IDF, character n-grams | **0.9344** | **0.9817** |
| TF-IDF, word n-grams | 0.8935 | 0.9752 |
| FastText mean vector | 0.8173 | 0.9416 |
| word2vec mean vector | 0.8244 | 0.9348 |
| GloVe mean vector | 0.6910 | 0.7847 |

Averaging blurs exactly the tokens spam detection runs on: phone numbers,
amounts, "txt", "claim". GloVe does worst because its meanings come from
general English. Its neighbours for `free` are *through, without, allowed*:
"free of". In SMS spam, `free` sits next to *ringtone*.

### Robustness: the prediction was wrong

The expectation was that FastText, which can build a vector for a mangled
word, would survive character attacks the way character TF-IDF did, while
word2vec collapsed. Spam recall at 50% perturbation:

| Attack | TF-IDF word | TF-IDF char | word2vec | FastText | GloVe |
|---|---|---|---|---|---|
| *clean* | 0.972 | 0.966 | 0.915 | 0.961 | 0.910 |
| zero-width | 0.644 | **0.966** | 0.565 | 0.763 | 0.113 |
| homoglyph | 0.870 | **0.961** | 0.904 | 0.887 | 0.814 |
| leetspeak | 0.864 | **0.977** | 0.910 | 0.910 | 0.887 |
| typo | 0.763 | **0.944** | 0.831 | 0.802 | 0.655 |
| spacing | 0.938 | **0.938** | 0.898 | 0.904 | 0.848 |

- **FastText beats word2vec only on zero-width evasion** (0.763 vs 0.565). On
  homoglyphs, leetspeak, typos and spacing, word2vec loses the same or less
  recall. The two share training data and vocabulary (their out-of-vocabulary
  rates are identical), so the subword component is the only difference.
- **The likely reason: a bad vector can be worse than no vector.** An attacked
  word is unknown to word2vec and simply dropped. The untouched tokens, often
  phone numbers and prize amounts that letter-substitution attacks cannot
  change, still say "spam". FastText does produce a vector for the mangled
  word, but it is a noisy guess: the homoglyph `frее` has cosine 0.73 to
  `free`, yet its nearest neighbours are *ride, download, outside*. Averaging
  that guess in dilutes the signal.
- **Character TF-IDF is the only representation robust to all five attacks.**

**A methodological note.** The first run appeared to confirm the prediction
(word2vec recall 0.057 under zero-width, 0.237 under spacing). That was a
confound: the embedding tokenizer kept single characters, scikit-learn's
TF-IDF tokenizer drops them, and these attacks shatter words into letters.
With the tokenizers aligned, the same numbers are 0.565 and 0.898.
Tokenization is part of the representation, so it has to be held fixed when
representations are compared.

### Corpus leakage with GloVe

The four evaluations from the headline result, repeated on GloVe vectors
(word2vec and FastText would need retraining for every held-out corpus to avoid
leakage, which is too slow on CPU):

| | TF-IDF word | GloVe |
|---|---|---|
| **A** corpus-identity probe | 0.919 | 0.764 |
| **B** pooled random split, F1 | 0.9873 | 0.9217 |
| **C** leave-one-corpus-out, mean F1 over the 4 mixed corpora | 0.871 | 0.757 |
| **C** `nazario` recall | 0.658 | 0.530 |
| **D** within-corpus F1 | 0.970 - 0.9965 | 0.906 - 0.979 |

GloVe vectors carry less corpus fingerprint (probe 0.76 vs 0.92, though still
4.6x chance). That did **not** make them generalise better: every
cross-corpus number is lower. The representation carries less of everything,
the phishing signal included.

**Overall:** on this task, bag-of-n-grams beats averaged word vectors on
accuracy, on robustness and on cross-corpus transfer. Averaging discards word
order and weights "the" as heavily as "claim". The natural next step is
contextual embeddings from a transformer, not a better average.

---

## Why there is no temporal evaluation

Training on old attacks and testing on new ones is the ideal evaluation for
this problem. It is not possible on either corpus, for different reasons, and
both are documented rather than worked around:

- **SMS Spam Collection has no timestamps at all.** Leakage-aware splitting is
  the partial substitute: it removes duplicate leakage but not concept drift.
- **The phishing archive has dates, but they are unusable.** Only 59.7% of
  messages carry one, and the `Date:` header is attacker-controlled — the
  archive contains years 0102, 2082 and 2100. After filtering to a plausible
  window, the deeper problem appears:

```
2002    5,622 msgs   28% phishing   <- SpamAssassin
2008   38,947 msgs   56% phishing   <- CEAS
every other year:   ~100% phishing
```

Ham exists in **two years only**. Any temporal cut yields a near-single-class
test set that a constant "phishing" prediction scores well on. **Date is
confounded with label.** `temporal_split()` ships with a warning docstring
explaining this; `scripts/08_check_dates.py` is the evidence.

---

## Fifth result: defences work on the attacks they were built for, and much less on the rest

The second result showed what breaks under evasion. This one tries to fix it,
then attacks the fix. Two kinds of defence (`src/defences.py`):

- **Input normalisation.** Strip invisible characters, fold Unicode (NFKC),
  map common Cyrillic and Greek lookalikes back to Latin, rejoin spaced-out
  words (`F R E E`, `c.l.i.c.k`), and turn digits back into letters only
  where a letter is expected (`fr33` → `free`, but `150p` and `0800` are left
  alone). Applied at training *and* prediction time.
- **Adversarial training.** Add one perturbed copy of every training message
  per attack, at a random strength. Ham is perturbed too, so the model cannot
  learn "odd characters mean spam".

A defence that is only tested on the attacks it was designed for is grading
its own homework, so there are two extra checks:

- **Leave-one-attack-out.** Train on four attacks, test on the fifth.
- **Unseen homoglyphs.** An attacker who has read the normaliser and uses
  lookalike characters it does not list (`ɑ ҽ ӏ օ ꜱ`, `homoglyph_unseen` in
  `src/perturb.py`). No defence ever trains on it.

Spam recall at 50% perturbation, SMS corpus, leakage-aware split.
"Held out" means adversarial training never saw that attack:

**`logreg_word`** (the model that collapsed in the second result)

| Attack | No defence | Normalise | Adv. training | Adv. training, held out |
|---|---|---|---|---|
| zero-width | 0.644 | **0.972** | 0.944 | 0.723 |
| homoglyph | 0.870 | **0.972** | 0.955 | 0.887 |
| leetspeak | 0.864 | 0.955 | 0.955 | 0.881 |
| typo | 0.763 | 0.763 | **0.949** | 0.763 |
| spacing | 0.938 | 0.966 | 0.960 | 0.955 |
| **unseen homoglyphs** | 0.870 | **0.870** | 0.927 | — |
| clean F1 / false positives | 0.894 / 36 | 0.894 / 36 | 0.860 / 51 | |

**`logreg_union`** (word + character features, the family the deployed SMS
model belongs to)

| Attack | No defence | Normalise + adv. training | Same, held out |
|---|---|---|---|
| zero-width | 0.921 | 0.966 | 0.966 |
| typo | 0.876 | **0.960** | 0.927 |
| spacing | 0.921 | 0.955 | 0.960 |
| unseen homoglyphs | 0.944 | 0.955 | — |
| clean F1 / false positives | 0.958 / 8 | 0.955 / 10 | |

Full tables for all three models and four defence settings are in
`results/defences_summary.csv`; the chart is `results/defences.png`.

What this shows:

- **Normalisation is precise and brittle.** It fully repairs zero-width and
  homoglyph evasion on the word model at no clean cost. Against the same
  trick with characters missing from its table it does *nothing*: 0.870
  before, 0.870 after. A rule list protects against the attacks its author
  thought of.
- **Adversarial training transfers only partly.** Trained on all five
  attacks, the word model recovers to 0.94-0.96 everywhere. With the scored
  attack held out, zero-width recall is 0.723 and typo recall does not move
  (0.763). Most of the gain is memorising specific tricks, not general
  robustness.
- **Adversarial training costs precision on ordinary messages.** On the word
  model, clean false positives ranged from 35 to 58 across five training
  seeds, against 36 undefended.
- **The cheapest defence was already in place.** Character n-grams
  (`logreg_char`) lose at most 0.03 recall to any attack, including unseen
  homoglyphs, and no defence improves them meaningfully. The best feature
  choice did more than either defence bolted onto a weak one.
- **On SMS, leetspeak is also how people write.** The normaliser changes 313
  of 5,574 *clean* messages: `gr8` → `grb`, `ni8` → `nib`, `MobileUpd8` →
  `MobileUpdb`. Textspeak and evasion use the same characters, so every
  de-leet rule damages some honest text. Its cost is small here (clean F1
  unchanged on the word model) but it is not zero, and it would differ on
  another corpus.

**How much the numbers can be trusted.** The test set has 177 spam
messages, so one message is 0.006 recall; differences under about 0.02 are
one to three messages and should not be read as a ranking. Adversarial
training depends on a random draw, so the three headline cases were rerun
with five seeds (`scripts/15_defence_seeds.py`,
`results/defence_seeds.csv`):

| Case | All attacks trained | Attack held out |
|---|---|---|
| word, adv. training, zero-width | 0.870-0.944 | 0.712-0.757 |
| word, adv. training, typo | 0.893-0.955 | 0.757-0.791 |
| union, normalise + adv. training, typo | 0.955-0.961 | 0.927-0.938 |

The held-out gap holds on every seed.

---

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/00_check_env.py

# SMS corpus - leakage and adversarial robustness
python scripts/01_download_data.py
python scripts/02_run_baselines.py       # split comparison
python scripts/03_robustness.py          # evasion attacks
python scripts/14_defences.py            # defences, then attack the defences
python scripts/15_defence_seeds.py       # seed sensitivity of those results
python scripts/04_error_analysis.py      # read the errors
python scripts/05_profile_data.py

# Email corpus - corpus leakage
#   download archive.zip from the Kaggle link below first
python scripts/06_extract_phishing.py [path/to/archive.zip]
python scripts/07_profile_corpora.py     # per-corpus profile
python scripts/08_check_dates.py         # date-quality audit
python scripts/09_corpus_generalisation.py   # A/B/C/D above

# Word embeddings - downloads GloVe (~130 MB) on first run
python scripts/13_embeddings.py          # sms | email | all (default)
```

Everything is deterministic (`SEED = 42` in `config.py`).

## Trained models

```bash
python scripts/11_train_final.py sms       # ~1 min  -> models/sms_classifier.joblib
python scripts/11_train_final.py email     # ~10 min -> models/email_classifier.joblib

python scripts/12_predict.py sms   --demo
python scripts/12_predict.py email --demo
python scripts/12_predict.py sms   "WINNER! Claim your prize, call 09061701461"
python scripts/12_predict.py email --file inbox.txt     # one message per line
```

Each model ships with a JSON card (`models/*.json`) recording its decision
threshold, validation metrics, and known limits. `--demo` runs four examples
clean, then again with invisible zero-width-character evasion applied.

| Model | Features | Size | Threshold | Validation |
|---|---|---|---|---|
| `sms_classifier` | word 1-2 + char 2-5 | 0.7 MB | 0.450 | F1 0.9658, leakage-aware split |
| `email_classifier` | word 1-2 + char 3-5, capped | 7.4 MB | 0.252 | F1 0.9841, **pooled - inflated** |

**The SMS model** classifies all four demo messages correctly, clean and under
evasion. Evasion lowers spam probability (0.979 → 0.829) without crossing the
threshold — the character-feature robustness from experiment 2, visible live.

**The email model has visible false positives in its own demo, and they are
left in deliberately.** Changing the examples or the threshold until the demo
passed would be tuning on the demo.

```
clean
  legitimate p=0.033   Meeting moved - the Thursday sync is moved to 3pm...
  PHISHING   p=0.998   Your account has been suspended - verify at http://...
  PHISHING   p=0.421   Invoice #4471 - attached invoice for consulting work   <- wrong
  PHISHING   p=0.889   URGENT: Confirm your password - IT department...

zero-width evasion
  PHISHING   p=0.341   Meeting moved - the Thursday sync is moved to 3pm...   <- wrong
```

Two causes, both consistent with the findings above:

- **The invoice.** "Please find attached the invoice… payment terms" is also
  the exact wording of invoice fraud and business-email compromise. On text
  alone the two are not separable; sender and header signals are needed, and
  this model ignores both. The low threshold (a false negative costs 10× a
  false positive) makes it quick to flag.
- **The meeting email under evasion.** On SMS, inserted invisible characters
  *lowered* the spam score. On email they *raise* the phishing score of a
  legitimate message. The likely reason is corpus leakage again: the phishing
  corpora carry encoding artifacts and odd characters, the ham corpora are
  mostly clean ASCII, so "unusual characters" became a phishing signal. The
  practical upside is that this attack does not evade the model; the downside
  is false positives on non-ASCII or copy-pasted legitimate mail.

The card's leave-one-corpus-out figures were measured on the word-only
`logreg_email` variant, not on the saved word+char model — indicative, not
exact.

## Layout

```
config.py            paths, seed, cost ratio, duplicate threshold
src/data.py          SMS loading; random vs leakage-aware splitting
src/corpora.py       email corpora; temporal + leave-one-corpus-out splits
src/perturb.py       five evasion attacks + one held-out homoglyph attack
src/defences.py      input normalisation; adversarial-training augmentation
src/models.py        six TF-IDF pipelines (word / char / union / email-scale)
src/evaluate.py      metrics, cost-sensitive thresholds, error dumps
src/embeddings.py    word2vec + FastText from scratch; GloVe loader
scripts/00..09_*.py  the experiments, in order
scripts/10_smoke_test.py    checks across every module
scripts/13_embeddings.py    experiment 5: word embeddings
scripts/14_defences.py      experiment 6: defences and attacks on them
scripts/15_defence_seeds.py experiment 6b: seed sensitivity
scripts/11_train_final.py   train + save deployable models
scripts/12_predict.py       classify new messages; --demo
models/              saved classifiers + JSON model cards
results/             generated CSV, JSON and plots
```

## Method notes

- **The pooled `phishing_email.csv` is deliberately unused.** It is all six
  corpora concatenated with provenance discarded. Once you cannot tell which
  corpus a message came from, you cannot run evaluation C — which is the only
  one that reveals the problem. `src/corpora.py` loads the per-source files.
- **Duplicate clustering.** Cosine on character 5-gram TF-IDF, union-find to
  merge transitively, threshold 0.80. Above 20,000 messages this becomes the
  dominant cost of the pipeline, so it falls back to normalised-text hashing —
  linear, but it under-counts. The fallback never invents clusters, so a
  grouped split built on it stays conservative.
- **Thresholds are tuned on validation, never on test.** A false negative is
  weighted 10x a false positive (`FN_COST_RATIO`).
- **Accuracy is not the headline.** The SMS corpus is 86.6% ham, so predicting
  "never spam" already scores 0.866.

## Limitations

- **The 52% phishing base rate is fiction.** Real mailflow is a few percent.
  `subsample_to_prior()` exists for this; the headline numbers above do not use
  it, so precision is flattered throughout.
- **Evaluation C is confounded in one direction.** Holding out `nazario` or
  `nigerian` leaves a single-class test set, so only recall is defined. Recall
  on unseen phishing is still the operationally meaningful number, but it is
  not comparable to the F1 values in the same table.
- **Attacks are non-adaptive.** A real attacker would query the model and
  search for the edit that flips it. The held-out homoglyph attack is a crude
  stand-in for an attacker who adapts to the defence; a query-based attack
  against the defended models is not evaluated.
- **Defences are evaluated on SMS only.** The email corpus has the corpus
  leakage problem from the first result, which would confound any robustness
  comparison there. Thresholds for defended models are picked on clean
  validation data, matching how the undefended models were tuned.
- **Label noise is present.** Manual error reading surfaced SMS messages
  labelled spam that read as ordinary forwarded jokes
  (`results/error_analysis.json`). Some residual error is the ceiling of the
  labels, not of the model.
- **No contextual models.** Averaged word embeddings did worse than TF-IDF
  everywhere. Whether a fine-tuned transformer closes the `nazario` gap is
  the natural next experiment.
- **The embedding comparison is deliberately simple.** word2vec and FastText
  were trained on ~3,300 SMS messages, which is tiny; a large pretrained
  FastText model (1 GB+) would be a fairer test of subwords. Plain mean
  pooling was used; TF-IDF-weighted averaging would likely do better. The
  from-scratch implementation is checked by `10_smoke_test.py` and by
  sensible nearest neighbours, but it has not been benchmarked against
  gensim, which cannot be installed here. All embedding results use a single
  random seed.

## Data

**SMS Spam Collection** — 5,574 messages, 13.4% spam.
Almeida, Hidalgo & Yamakami (2011), UCI Machine Learning Repository.

**Phishing email collection** — 82,486 emails across six corpora, 52.0%
phishing. Distributed via
[Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset).
Primary sources: Enron-Spam (Metsis, Androutsopoulos & Paliouras, 2006), the
Nazario phishing corpus, CEAS 2008, SpamAssassin public corpus, Ling-Spam, and
a Nigerian fraud collection. Cite the primary corpora, not the aggregate.

**Embeddings.** GloVe: Pennington, Socher & Manning (2014), 100d
Wikipedia + Gigaword vectors via the gensim-data release. word2vec: Mikolov et
al. (2013). FastText: Bojanowski, Grave, Joulin & Mikolov (2017).
