# -*- coding: utf-8 -*-
"""Build a Danbooru tag -> tag-group mapping from wiki dumps.

Two sources, merged:
1. deepghs/danbooru_wikis_full wiki_pages.csv -- the tag_group:* pages
   themselves; every [[tag]] link in a group page marks membership.
2. isek-ai/danbooru-wiki-2024 parquet -- per-tag wikis whose "See also"
   sections link [[Tag Group:X]]; the reverse direction.

Output: tag_groups.json {tag: [group, ...]} restricted to the
co-occurrence vocabulary (tag_cooc.npz), written next to this script.

Run with the ComfyUI venv python (needs pyarrow + the HF downloads under
~/workspace/danbooru_dumps/wiki/; see hf_hub_download calls below).
"""
import csv
import json
import os
import re
import sys

DUMP_DIR = os.path.expanduser("~/workspace/danbooru_dumps/wiki")
HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "..", "nodes", "lib")
sys.path.insert(0, LIB)
from tag_guard import _load_cooc  # noqa: E402

LINK = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]*)?\]\]")
GROUP_LINK = re.compile(
    r"\[\[\s*tag[ _]group\s*:\s*([^\]|]+?)\s*(?:\|[^\]]*)?\]\]", re.I)
NON_TAG_PREFIXES = (
    "tag group:", "tag_group:", "list of", "pool #", "post #",
    "howto:", "about:", "help:", "topic #", "forum #",
)


def norm(s):
    return s.strip().lower().replace("_", " ")


def from_group_pages(mapping):
    """deepghs group pages: [[tag]] links inside tag_group:* bodies."""
    path = os.path.join(DUMP_DIR, "wiki_pages.csv")
    csv.field_size_limit(sys.maxsize)
    with open(path) as f:
        for row in csv.DictReader(f):
            title = row["title"].lower()
            if not title.startswith("tag_group:") or row["is_deleted"] == "True":
                continue
            group = norm(title[len("tag_group:"):])
            for m in LINK.findall(row["body"]):
                t = norm(m)
                if t.startswith(NON_TAG_PREFIXES):
                    continue
                mapping.setdefault(t, set()).add(group)


def from_tag_wikis(mapping):
    """isek-ai per-tag wikis: [[Tag Group:X]] links in general tag pages."""
    import pyarrow.parquet as pq
    path = os.path.join(DUMP_DIR, "data", "train-00000-of-00001.parquet")
    t = pq.read_table(path, columns=["title", "body", "category"]).to_pydict()
    for title, body, cat in zip(t["title"], t["body"], t["category"]):
        if cat != "general" or not body:
            continue
        for g in GROUP_LINK.findall(body):
            mapping.setdefault(norm(title), set()).add(norm(g))


def main():
    vocab = set(_load_cooc()["tags"])
    mapping = {}
    from_group_pages(mapping)
    from_tag_wikis(mapping)
    out = {t: sorted(gs) for t, gs in mapping.items() if t in vocab}
    dst = os.path.join(HERE, "..", "..", "resources", "tag_groups.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=0, sort_keys=True, ensure_ascii=False)
    print("vocab %d, mapped %d (%.1f%%) -> %s"
          % (len(vocab), len(out), 100.0 * len(out) / len(vocab), dst))


if __name__ == "__main__":
    main()
