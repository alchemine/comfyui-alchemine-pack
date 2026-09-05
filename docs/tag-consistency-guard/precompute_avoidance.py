# -*- coding: utf-8 -*-
"""Precompute the tag avoidance table TagSuggest vetoes with.

The suggest artifact stores a fixed 512 repulsion neighbours per tag,
which sounds generous and is not: the rows saturate, so "absent from the
table" stops meaning "not avoided" and starts meaning "not among the 512
most avoided". For bar_(place) the row bottoms out at lift 0.81, so a
pair like bar_(place)/computer_keyboard -- 0 posts out of 4.8M -- is
simply invisible at runtime, and the sampler happily puts a keyboard in
a bar.

This table replaces the fixed cap with a significance test, so a tag
keeps every avoidance the corpus can actually demonstrate and nothing
more: 35 pairs for the median tag, 10k for the ones that genuinely
contradict half the vocabulary.

The test is the Poisson lower tail. Under independence a pair is seen
Poisson(expected) times, expected = c_i * c_j / n_posts, so observing
`observed` or fewer has probability p; small p means the corpus is
avoiding the pair rather than just short of examples. p is stored
quantised (see SIG_SCALE) and the runtime picks its own alpha, which is
what makes the artifact tunable without a rebuild.

Pairs are stored both ways round so a lookup is one slice.

Source: the full solo-post co-occurrence matrix built by
precompute_conflicts.py (cooc_C_solo_min100.npy). Solo posts only, for
the reason given there: a 2girls post says nothing about one character.

    python precompute_avoidance.py [--alpha 0.05]
"""
import argparse
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "resources")
DUMP_DIR = os.path.expanduser("~/workspace/danbooru_dumps")
C_PATH = os.path.join(DUMP_DIR, "cooc_C_solo_min100.npy")
META_PATH = os.path.join(DUMP_DIR, "cooc_C_solo_min100_meta.npy")
OUT = os.path.join(RES, "avoidance_v1.npz")

# quantisation for -log10(p): 20 steps per decade keeps alpha resolution
# far finer than anyone will tune it, in one byte up to p = 1e-12
SIG_SCALE = 20.0

# below this many expected co-occurrences no observation can be
# significant at any useful alpha, so the pair is skipped outright --
# it also keeps the chunk masks sparse
MIN_EXPECTED = 1.0

# Significance alone is the wrong test and picks the wrong pairs. With a
# large expected count even a mild shortfall clears any alpha, so
# bar_(place) "significantly avoids" skirt, navel and closed_eyes --
# true as a frequency statement, useless as a contradiction. A pair has
# to be both unlikely to be chance and actually lopsided, so the corpus
# must have delivered less than this fraction of the expected count.
MAX_LIFT = 0.35


def build_vocab():
    """The judged vocabulary, matching precompute_conflicts.build_vocab.

    Row order has to agree with the matrix exactly, and the matrix was
    built over the space form sorted as such -- "computer keyboard", not
    "computer_keyboard". Sorting the underscore form instead silently
    permutes part of the vocabulary, which does not change the shape and
    so fails only as nonsense statistics.
    """
    vocab = []
    with open(os.path.join(HERE, "danbooru_general_tags.csv")) as f:
        next(f)                   # header
        for name, count in csv.reader(f):
            if int(count) >= 100:
                vocab.append(name.replace("_", " "))
    return sorted(vocab)


def main(alpha, chunk):
    vocab = build_vocab()
    n = len(vocab)
    C = np.load(C_PATH, mmap_mode="r")
    if C.shape != (n, n):
        raise SystemExit("matrix %s does not match vocabulary %d"
                         % (C.shape, n))
    posts = float(np.load(META_PATH, allow_pickle=True)[0])
    counts = np.diag(C).astype(np.float64)
    print("vocab %d, posts %d, alpha %g" % (n, posts, alpha))

    cut = -np.log10(alpha)
    rows_ids = [[] for _ in range(n)]
    rows_sig = [[] for _ in range(n)]
    total = 0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        obs = np.asarray(C[start:end], dtype=np.float64)
        expected = np.outer(counts[start:end], counts) / posts
        # Poisson lower tail. observed == 0 is exact (p = e^-E); above it
        # the normal approximation is close enough and vastly cheaper,
        # and it is only consulted where E >= 5 so the approximation
        # holds.
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (obs - expected) / np.sqrt(np.maximum(expected, 1e-9))
        logp = np.where(
            obs == 0,
            expected / np.log(10.0),                    # -log10(e^-E)
            np.where((expected >= 5.0) & (z < 0),
                     -np.log10(np.maximum(_norm_cdf(z), 1e-13)),
                     0.0),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(expected > 0, obs / expected, 1.0)
        keep = ((expected >= MIN_EXPECTED) & (logp >= cut)
                & (lift < MAX_LIFT))
        keep[np.arange(end - start), np.arange(start, end)] = False
        sig = np.minimum(np.round(logp * SIG_SCALE), 255).astype(np.uint8)
        for local in range(end - start):
            j = np.nonzero(keep[local])[0]
            if not len(j):
                continue
            i = start + local
            s = sig[local][j]
            rows_ids[i].append(j.astype(np.int32))
            rows_sig[i].append(s)
            total += len(j)
        print("  rows %d-%d, %d pairs so far" % (start, end, total))

    indptr = np.zeros(n + 1, dtype=np.int64)
    ids, sigs = [], []
    for i in range(n):
        if rows_ids[i]:
            ids.append(np.concatenate(rows_ids[i]))
            sigs.append(np.concatenate(rows_sig[i]))
            indptr[i + 1] = indptr[i] + len(ids[-1])
        else:
            indptr[i + 1] = indptr[i]
    ids = np.concatenate(ids) if ids else np.zeros(0, np.int32)
    sigs = np.concatenate(sigs) if sigs else np.zeros(0, np.uint8)
    np.savez_compressed(OUT, tags=np.array(vocab), indptr=indptr,
                        ids=ids, sig=sigs, posts=np.array([posts]),
                        sig_scale=np.array([SIG_SCALE]))
    print("saved %s: %d directed pairs, %.1f MB"
          % (OUT, len(ids), os.path.getsize(OUT) / 1e6))


def _norm_cdf(z):
    from scipy.special import ndtr
    return ndtr(z)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=512)
    main(*vars(ap.parse_args()).values())
