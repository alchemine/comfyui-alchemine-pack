# -*- coding: utf-8 -*-
"""Precompute a data-driven tag conflict table from Danbooru post dumps.

Vocabulary: all live general tags with post_count >= 100 (~21k tags),
no category restriction.

Only posts tagged 'solo' are counted: multi-character posts inflate
co-occurrence of per-character attributes (blonde hair + red hair on a
2girls post says nothing about one character wearing both).

Pipeline:
    1. Full pairwise co-occurrence matrix C (X^T X over 8M posts).
    2. Profile similarity: cosine between co-occurrence rows (PPMI-weighted).
       High cosine = tags used in the same context (substitutes).
       Profile columns are restricted to frequent tags (post_count >=
       CTX_MIN_COUNT, the old >=1000 vocabulary): rare context dimensions
       carry noisy PMI values that deflate every cosine and would shift
       the calibrated 0.75 runtime threshold. Rows (judged tags) still
       cover the full >=100 vocabulary.
    3. Direct co-occurrence, chance-normalized:
       lift = C[i,j] * n_posts / (C[i,i] * C[j,j]).
       lift << 1 = the tags avoid each other (conflict evidence);
       lift >= 1 = they attract each other. Pairs whose expected count
       is below E_MIN carry no avoidance evidence (a rare pair expected
       ~2 times observing 0 is chance, not avoidance), so their stored
       lift is floored at 1.0 (neutral) and they are never conflict
       candidates. At E >= 15 the chance of observing 0 is ~3e-7, so
       raw lift is trustworthy. This is what makes the >=100 vocabulary
       safe. An additive prior ((C+3)/E) was tried instead and rejected:
       it erases genuine avoidance at moderate expected counts
       (foot focus/narrow waist: C=5, E=26, a user-judged conflict).
       Absolute overlap
       C[i,j]/min(...) is kept for reporting but NOT used for judgment:
       it penalizes pairs with popular partners at chance level
       (smug/pink eyes ov=0.04 is ~half of chance, not avoidance).
    4. For each tag keep every conflict-candidate pair (cosine >= COS_KEEP
       and overlap < OV_KEEP, generous margins around runtime thresholds)
       plus top-K by overlap (compatibility evidence), storing
       (cosine, overlap) per pair. A plain top-K-by-cosine does NOT work:
       popular tags' top slots fill up with co-occurring companions
       (day <-> blue sky 0.98) and push out substitutes (night 0.93).

Conflict at runtime: cosine >= cos_th AND overlap < ov_th (thresholds are
applied at runtime so they stay tunable without recomputation).

Writes nodes/resources/tag_cooc.npz:
    tags: (N,) names (spaces)   counts: (N,) uint32   n_posts: uint32
    nbr_indptr: (N+1,)          nbr_ids: (nnz,) int32
    nbr_cos: (nnz,) float16     nbr_ov: (nnz,) float16
"""
import csv
import glob
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).parent
LIB = HERE.parent.parent / "nodes" / "lib"
RES = HERE.parent.parent / "nodes" / "resources"

MIN_COUNT = 100
CTX_MIN_COUNT = 1000  # profile (PPMI context) columns: frequent tags only
# Cell-level confidence weighting (PPMI * C/(C+PMI_CONF)) over the full
# vocab columns was tried as a principled replacement for the hard
# CTX_MIN_COUNT gate and REJECTED: lambda=5 scored 17/23 on
# calibration_pairs vs 21/23 for the hard gate. The shrinkage also
# rescales legitimate moderate-count cells, so every profile moves and
# the cos_th=0.75 calibration breaks; separation got worse, not just
# shifted (compatible legs apart/sitting 0.73 > conflict wading/sitting
# 0.61). Any change to the profile geometry requires re-calibrating.
PMI_CONF = 0.0        # 0 = disabled
TOP_K = 48       # top-K by overlap (compatibility evidence)
COS_KEEP = 0.6   # keep all pairs with cos >= this ...
LIFT_KEEP = 1.0  # ... and lift < this (conflict candidates)
E_MIN = 15.0     # min expected pair count for avoidance evidence
BATCH = 65536
DUMP_DIR = "/home/dev/workspace/danbooru_dumps"
# int32 C matrix + tiny meta file, keyed by vocab threshold so the old
# >=1000 cache stays usable for comparison runs
C_CACHE = DUMP_DIR + "/cooc_C_solo_min%d.npy" % MIN_COUNT
C_META = DUMP_DIR + "/cooc_C_solo_min%d_meta.npy" % MIN_COUNT


def build_vocab():
    """Return (vocab, ctx_tags): full judged vocabulary and the subset
    used as PPMI profile dimensions."""
    vocab, ctx = [], set()
    with open(HERE / "danbooru_general_tags.csv") as f:
        next(f)
        for name, count in csv.reader(f):
            if int(count) >= MIN_COUNT:
                t = name.replace("_", " ")
                vocab.append(t)
                if int(count) >= CTX_MIN_COUNT:
                    ctx.add(t)
    return sorted(vocab), ctx


def main():
    import pyarrow.parquet as pq

    tags, ctx_tags = build_vocab()
    index = {t: i for i, t in enumerate(tags)}
    n = len(tags)
    ctx_ids = np.array([i for i, t in enumerate(tags) if t in ctx_tags],
                       dtype=np.int64)
    ctx_pos = {int(i): p for p, i in enumerate(ctx_ids)}  # vocab idx -> P col
    n_ctx = len(ctx_ids)
    print("vocab: %d tags (%d context dims)" % (n, n_ctx))

    n_posts = 0
    if Path(C_CACHE).exists():
        # memmap: keep the 1.8GB matrix on disk, page in rows as needed
        C = np.load(C_CACHE, mmap_mode="r")
        n_posts = int(np.load(C_META)[0])
        print("loaded cached C (posts: %d)" % n_posts)
        run_count = False
    else:
        # int32 keeps the dense accumulator at ~1.8GB for 21k tags
        # (pair counts are bounded by n_posts ~ 4.8M, well within int32)
        C = np.zeros((n, n), dtype=np.int32)
        run_count = True

    shards = sorted(glob.glob(DUMP_DIR + "/db*.parquet")) if run_count else []
    assert shards or not run_count, "no shards in " + DUMP_DIR
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(columns=["general", "is_deleted"],
                                     batch_size=BATCH):
            gens = batch.column("general").to_pylist()
            dels = batch.column("is_deleted").to_pylist()
            indptr = [0]
            indices = []
            for gen, dele in zip(gens, dels):
                if not dele and gen:
                    tags_in_post = gen.split(", ")
                    if "solo" in tags_in_post:
                        n_posts += 1
                        row = {index[t] for t in tags_in_post if t in index}
                        indices.extend(row)
                indptr.append(len(indices))
            if not indices:
                continue
            X = sparse.csr_matrix(
                (np.ones(len(indices), dtype=np.int32),
                 np.array(indices, dtype=np.int32),
                 np.array(indptr, dtype=np.int64)),
                shape=(len(gens), n),
            )
            # keep the product sparse: a dense .toarray() would allocate
            # another n^2 int64 (~3.7GB) per batch
            S = (X.T @ X).tocoo()
            C[S.row, S.col] += S.data.astype(np.int32)
        print("done %s (posts: %d)" % (shard, n_posts))

    if run_count:
        np.save(C_CACHE, C)
        np.save(C_META, np.array([n_posts], dtype=np.int64))
        # reopen as memmap so the distillation below doesn't hold the
        # full matrix in RAM on top of the profile matrix P
        del C
        C = np.load(C_CACHE, mmap_mode="r")

    counts = np.ascontiguousarray(np.diag(C)).astype(np.float64)
    ctx_counts = counts[ctx_ids]
    total = ctx_counts.sum()

    # --- PPMI-normalized profile matrix (row blocks, float32) --------------
    # rows: full vocab; columns: frequent context tags only
    BLOCK = 512
    P = np.empty((n, n_ctx), dtype=np.float32)  # ~730MB at 21k x 8.5k
    for lo in range(0, n, BLOCK):
        hi = min(lo + BLOCK, n)
        Cf = C[lo:hi][:, ctx_ids].astype(np.float64)
        for r in range(lo, hi):
            p = ctx_pos.get(r)
            if p is not None:
                Cf[r - lo, p] = 0.0
        expected = np.outer(counts[lo:hi], ctx_counts) / total
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(Cf / np.maximum(expected, 1e-9))
        P[lo:hi] = np.maximum(pmi, 0.0)
        if PMI_CONF > 0:
            P[lo:hi] *= (Cf / (Cf + PMI_CONF))
    norms = np.linalg.norm(P, axis=1)
    P /= np.maximum(norms, 1e-9)[:, None]

    # --- distill conflict candidates (row blocks) --------------------------
    print("computing cosine similarity...")
    nbr_ids, nbr_cos, nbr_ov, nbr_lift, indptr = [], [], [], [], [0]
    for lo in range(0, n, BLOCK):
        hi = min(lo + BLOCK, n)
        COS_blk = P[lo:hi] @ P.T
        Cb = C[lo:hi].astype(np.float64)
        mins = np.minimum.outer(counts[lo:hi], counts)
        OV_blk = (Cb / np.maximum(mins, 1)).astype(np.float32)
        expected = np.outer(counts[lo:hi], counts) / n_posts
        LIFT_blk = (Cb / np.maximum(expected, 1e-9)).astype(np.float32)
        # pairs with too little expected mass can't prove avoidance:
        # floor their lift at neutral 1.0 (see module docstring)
        LIFT_blk = np.where(expected >= E_MIN, LIFT_blk,
                            np.maximum(LIFT_blk, 1.0))
        for i in range(lo, hi):
            cos_row, ov_row = COS_blk[i - lo], OV_blk[i - lo]
            lift_row = LIFT_blk[i - lo]
            ov_row[i] = 1.0
            lift_row[i] = 1.0
            candidates = np.where((cos_row >= COS_KEEP) & (lift_row < LIFT_KEEP))[0]
            top_ov = np.argpartition(-ov_row, TOP_K + 1)[: TOP_K + 1]
            nbrs = sorted(set(candidates.tolist() + top_ov.tolist()) - {i})
            nbr_ids.extend(nbrs)
            nbr_cos.extend(cos_row[nbrs].tolist())
            nbr_ov.extend(ov_row[nbrs].tolist())
            nbr_lift.extend(np.minimum(lift_row[nbrs], 60000.0).tolist())
            indptr.append(len(nbr_ids))

    np.savez_compressed(
        RES / "tag_cooc.npz",
        tags=np.array(tags),
        counts=counts.astype(np.uint32),
        n_posts=np.uint32(n_posts),
        nbr_indptr=np.array(indptr, dtype=np.int64),
        nbr_ids=np.array(nbr_ids, dtype=np.int32),
        nbr_cos=np.array(nbr_cos, dtype=np.float16),
        nbr_ov=np.array(nbr_ov, dtype=np.float16),
        nbr_lift=np.array(nbr_lift, dtype=np.float16),
    )
    print("saved %s (%d pairs)" % (RES / "tag_cooc.npz", len(nbr_ids)))

    # sanity samples (looked up from the distilled table)
    indptr_a = np.array(indptr, dtype=np.int64)
    ids_a = np.array(nbr_ids, dtype=np.int32)

    def show(a, b):
        i, j = index.get(a), index.get(b)
        if i is None or j is None:
            print("  %s / %s: not in vocab" % (a, b))
            return
        lo_, hi_ = indptr_a[i], indptr_a[i + 1]
        pos = np.nonzero(ids_a[lo_:hi_] == j)[0]
        if pos.size:
            k = int(lo_ + pos[0])
            print("  %s / %s: cos=%.3f ov=%.4f lift=%.3f"
                  % (a, b, nbr_cos[k], nbr_ov[k], nbr_lift[k]))
        else:
            print("  %s / %s: pair not stored (no conflict candidate)" % (a, b))

    for a, b in [("pond", "waterfall"), ("bikini", "dress"), ("day", "night"),
                 ("dark background", "pond"), ("ponytail", "twintails"),
                 ("blonde hair", "red hair"), ("smile", "pond"),
                 ("smug", "pink eyes"), ("night", "bikini"),
                 ("sitting", "standing"), ("indoors", "outdoors"),
                 ("nude", "dress"), ("wet hair", "pink eyes")]:
        show(a, b)


if __name__ == "__main__":
    main()
