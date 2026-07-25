# -*- coding: utf-8 -*-
"""Evaluate tag_cooc.npz against calibration_pairs.txt.

For each judged pair, applies the same decision rule as
tag_guard.is_conflict (with the second tag as the fixed ref) and
reports mismatches. Run after every precompute_conflicts.py rebuild.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "nodes", "lib"))
import tag_guard  # noqa: E402


def main():
    ok = miss = 0
    rows = []
    with open(os.path.join(HERE, "calibration_pairs.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            verdict, a, b = line.split("\t")
            ref = tag_guard.is_conflict(a, [b])
            got = "conflict" if ref else "compatible"
            stats = tag_guard.pair_stats(a, b)
            detail = ("cos=%.3f ov=%.3f lift=%.3f" % stats if stats
                      else "pair not stored")
            mark = "OK  " if got == verdict else "MISS"
            if got == verdict:
                ok += 1
            else:
                miss += 1
            rows.append("%s %-10s (got %-10s) %s / %s  [%s]"
                        % (mark, verdict, got, a, b, detail))
    print("\n".join(rows))
    print("\n%d/%d correct (%d miss)" % (ok, ok + miss, miss))
    return 1 if miss else 0


if __name__ == "__main__":
    sys.exit(main())
