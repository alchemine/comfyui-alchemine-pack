# -*- coding: utf-8 -*-
"""Precompute the subject-conjunction veto table.

Every table the sampler has is pairwise, and the sampler scores a
candidate by summing log lift over the context -- naive Bayes, a product
of pairwise terms. For the subject count that is wrong in a way more
data cannot fix, because the contradiction lives only in the
conjunction:

    lift(threesome | 1girl) = 0.59      near neutral
    lift(threesome | 1boy)  = 2.62      attracts, if anything
    product                 = 1.56      passes any veto
    lift(threesome | both)  = 0.02      52 posts where 2,636 were due

"1girl" does not mean one person, it means one girl, so 1girl+2boys
threesomes are ordinary and both pairwise terms are honest about their
own halves. Only together do they say three cannot be two.

Subject tags are few (26 in the vocabulary) and combine into a short
head of common pairs, so this table is small where the general case --
every pair of tags -- would not be.

Counting is the slow part (~4 minutes over 9.2M posts) and the
thresholds are the part worth arguing about, so the raw counts are
cached and a threshold sweep reads them back in seconds. --half builds
from even or odd post ids, which is how the thresholds get judged: a
veto that is a real contradiction appears in both halves, a veto that is
one tag being uncommon in that kind of picture does not.

Source: the 2026 metadata dump (9.23M posts), NOT the solo-only
co-occurrence matrix the other tables use -- a table about what two
subject tags do together cannot be built from posts that have one.

    python precompute_subject_joint.py [--alpha 1e-3] [--min-posts 2000]
    python precompute_subject_joint.py --half even --no-write
"""
import argparse
import os
import re
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "resources")
DUMP = os.path.join(HERE, "..", "..", "playground", "data",
                    "danbooru-2026-clean-metadata",
                    "danbooru2026_clean.parquet")
SUGGEST = os.path.join(RES, "suggest_v1.1.npz")
OUT = os.path.join(RES, "subject_joint_v1.npz")

# matches tag_veto.is_subject; duplicated so the script runs standalone
# against a dump, but the two must agree or the runtime looks up pairs
# the table was never built for
SUBJECT_RE = re.compile(r"^\d+\+?(boy|girl|other)s?$")
SUBJECT_TAGS = {"solo", "solo_focus", "male_focus", "female_focus",
                "multiple_boys", "multiple_girls", "multiple_others",
                "no_humans", "everyone"}

SIG_SCALE = 20.0        # as in precompute_avoidance: 20 steps per decade

# Defaults, all three overridable so a sweep costs nothing.
#
# MIN_EXPECTED guards the tail: below this many expected co-occurrences
# the corpus has no sample to judge the pair with, the way E_MIN does
# for tag_cooc. Observing 0 where 6 were due is unremarkable; observing
# 52 where 2,636 were due is not.
#
# The value comes from a split-half build (--half even against --half
# odd): a real contradiction shows up in both halves, an accident does
# not. At the original 5 only 53% of entries replicated. At 100, 91% do
# and the table drops from 7,664 entries to 907, with every entry that
# matters kept -- threesome and group_sex for 1girl+1boy, the female
# attributes for 1boy+solo, the male ones for 1girl+solo.
#
# The entries this drops were measured to change nothing: across
# thresholds 5 to 1000 the generated output was identical (0 subject
# contradictions, same 186 distinct tags over 30 draws), because the
# tags being dropped are too rare to be candidates in the first place.
# So this is hygiene rather than behaviour -- but an artifact whose
# entries half fail to replicate is not one to reason from later.
#
# MAX_SURPRISE is the one that matters. This table exists for the one
# thing the pairwise tables cannot say, so it should hold only that:
# compare the joint lift against the product of the pairwise lifts --
# the sampler's own naive-Bayes estimate -- and keep the tag only when
# the conjunction falls far below it. shrimp tempura is already
# predicted to be rare by both halves and lands near 1; threesome is
# predicted at 1.56 and observed at 0.02.
#
# MIN_PRODUCT drops what naive Bayes would have vetoed anyway: yuri
# against 1girl+1boy needs no entry here, the pairwise table has it.
DEFAULTS = dict(min_expected=100.0, max_lift=0.35, max_surprise=0.2,
                min_product=0.1)


def is_subject(tag):
    return tag in SUBJECT_TAGS or bool(SUBJECT_RE.match(tag))


def _norm_cdf(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def vocabulary():
    """The suggest vocabulary, which the runtime masks over.

    Rebuilding the list from the tag csv would look identical and be
    silently permuted, so it is read from the artifact it must match.
    """
    data = np.load(SUGGEST)
    return [str(t) for t in data["tags"]]


def scan(cache, half, min_posts, batch_size):
    """Count the dump into `cache`: marginals, singles, pair joints."""
    vocab = vocabulary()
    index = {t: i for i, t in enumerate(vocab)}
    subjects = sorted(t for t in vocab if is_subject(t))
    subject_slot = {t: i for i, t in enumerate(subjects)}
    pair_id, ordered = {}, []
    for a in range(len(subjects)):
        for b in range(a + 1, len(subjects)):
            pair_id[(subjects[a], subjects[b])] = len(ordered)
            ordered.append((a, b))
    print("vocab %d, subject tags %d, pairs %d"
          % (len(vocab), len(subjects), len(ordered)))

    def rows():
        f = pq.ParquetFile(DUMP)
        for batch in f.iter_batches(batch_size=batch_size,
                                    columns=["id", "tag_string_general"]):
            ids = batch.column(0).to_pylist()
            strs = batch.column(1).to_pylist()
            for pid, s in zip(ids, strs):
                if half == "even" and pid % 2:
                    continue
                if half == "odd" and not pid % 2:
                    continue
                yield s

    marg = np.zeros(len(vocab), dtype=np.int64)
    pair_posts = np.zeros(len(ordered), dtype=np.int64)
    subject_posts = np.zeros(len(subjects), dtype=np.int64)
    n_posts = 0
    t0 = time.time()
    for s in rows():                       # pass 1: what is worth a row
        n_posts += 1
        if not s:
            continue
        tags = s.split()
        ids, subj = [], []
        for t in tags:
            i = index.get(t)
            if i is not None:
                ids.append(i)
                if is_subject(t):
                    subj.append(t)
        marg[ids] += 1
        for t in subj:
            subject_posts[subject_slot[t]] += 1
        subj.sort()
        for x in range(len(subj)):
            for y in range(x + 1, len(subj)):
                pair_posts[pair_id[(subj[x], subj[y])]] += 1
    print("pass 1: %d posts, %.0fs" % (n_posts, time.time() - t0))

    kept = np.nonzero(pair_posts >= min_posts)[0]
    slot = {int(p): k for k, p in enumerate(kept)}
    print("pairs with >= %d posts: %d" % (min_posts, len(kept)))

    joint = np.zeros((len(kept), len(vocab)), dtype=np.int32)
    single = np.zeros((len(subjects), len(vocab)), dtype=np.int32)
    t0 = time.time()
    for s in rows():                       # pass 2: the counts themselves
        if not s:
            continue
        ids, subj = [], []
        for t in s.split():
            i = index.get(t)
            if i is not None:
                ids.append(i)
                if is_subject(t):
                    subj.append(t)
        if not ids or not subj:
            continue
        row = np.asarray(ids)
        for t in subj:
            single[subject_slot[t], row] += 1
        subj.sort()
        for x in range(len(subj)):
            for y in range(x + 1, len(subj)):
                k = slot.get(pair_id[(subj[x], subj[y])])
                if k is not None:
                    joint[k, row] += 1
    print("pass 2: %.0fs" % (time.time() - t0))

    np.savez_compressed(
        cache, tags=np.array(vocab), subjects=np.array(subjects),
        marg=marg, single=single, joint=joint,
        pair_a=np.array([ordered[p][0] for p in kept], dtype=np.int32),
        pair_b=np.array([ordered[p][1] for p in kept], dtype=np.int32),
        pair_posts=pair_posts[kept], subject_posts=subject_posts,
        n_posts=np.int64(n_posts))
    print("cached %s (%.1f MB)"
          % (os.path.basename(cache), os.path.getsize(cache) / 1e6))


def select(c, alpha, **th):
    """Apply the thresholds to cached counts -> per-pair veto lists."""
    p = dict(DEFAULTS, **{k: v for k, v in th.items() if v is not None})
    counts = c["marg"].astype(np.float64)
    n_posts = float(c["n_posts"])
    subject_posts = c["subject_posts"].astype(np.float64)
    cut = -np.log10(alpha)

    def pairwise(sub):
        exp = counts * subject_posts[sub] / n_posts
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(exp > 0, c["single"][sub] / exp, 1.0)

    out = []
    for k in range(len(c["pair_a"])):
        n_pair = float(c["pair_posts"][k])
        expected = counts * n_pair / n_posts
        obs = c["joint"][k].astype(np.float64)
        product = pairwise(c["pair_a"][k]) * pairwise(c["pair_b"][k])
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (obs - expected) / np.sqrt(np.maximum(expected, 1e-9))
            lift = np.where(expected > 0, obs / expected, 1.0)
            surprise = np.where(product > 0, lift / product, 1.0)
        logp = np.where(
            obs == 0, expected / np.log(10.0),
            np.where((expected >= 5.0) & (z < 0),
                     -np.log10(np.maximum(_norm_cdf(z), 1e-13)), 0.0))
        hit = np.nonzero((expected >= p["min_expected"]) & (logp >= cut)
                         & (lift < p["max_lift"])
                         & (surprise < p["max_surprise"])
                         & (product >= p["min_product"]))[0]
        out.append((hit.astype(np.int32),
                    np.minimum(np.round(logp[hit] * SIG_SCALE),
                               255).astype(np.uint8)))
    return out


def main(args):
    cache = os.path.join(
        HERE, "subject_joint_counts%s.npz"
        % ("" if args.half == "all" else "_" + args.half))
    if not os.path.exists(cache):
        scan(cache, args.half, args.min_posts, args.batch_size)
    c = np.load(cache, allow_pickle=False)
    picks = select(c, args.alpha, min_expected=args.min_expected,
                   max_lift=args.max_lift, max_surprise=args.max_surprise,
                   min_product=args.min_product)
    total = sum(len(h) for h, _ in picks)
    print("vetoes: %d over %d pairs" % (total, len(picks)))
    if args.no_write:
        return
    np.savez_compressed(
        OUT, tags=c["tags"], subjects=c["subjects"],
        pair_a=c["pair_a"], pair_b=c["pair_b"], pair_posts=c["pair_posts"],
        veto_pair=np.concatenate([np.full(len(h), k, np.int32)
                                  for k, (h, _) in enumerate(picks)]),
        veto_tag=np.concatenate([h for h, _ in picks]),
        veto_sig=np.concatenate([s for _, s in picks]),
        sig_scale=np.float64(SIG_SCALE), n_posts=c["n_posts"])
    print("wrote %s (%.1f MB)" % (OUT, os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--min-posts", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=100_000)
    ap.add_argument("--half", choices=("all", "even", "odd"), default="all")
    ap.add_argument("--min-expected", type=float)
    ap.add_argument("--max-lift", type=float)
    ap.add_argument("--max-surprise", type=float)
    ap.add_argument("--min-product", type=float)
    ap.add_argument("--no-write", action="store_true")
    main(ap.parse_args())
