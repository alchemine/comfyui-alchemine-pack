# -*- coding: utf-8 -*-
"""Category and rating labels for the tag vocabulary.

Four JSON layers under resources/group/ answer, for any tag, "which
knob does this belong to" and "how explicit is it":

    tags_v1.0.json        tag  -> [danbooru wiki group, ...]
    hierarchy_v1.0.json   the wiki's group tree, plus the groups its
                          table of contents omits and five of our own
    categories_v1.0.json  which tree nodes each user-facing category
                          owns, and the order that resolves a tag
                          whose groups span several
    ratings_v1.0.json     tag  -> g/s/q/e, from rating-tier statistics

A tag's category is the first category in `priority` order owning any
of its groups; a tag with no known group falls to the last category.
Ratings are cumulative, so a level is a ceiling: asking for "s" admits
g and s tags.
"""
import json
import os

RATING_ORDER = ("g", "s", "q", "e")

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "resources", "group")
_LABELS = None          # lazy singleton: Labels, or False if unavailable


def normalize(tag):
    return tag.strip().lower().replace("_", " ")


class Labels:
    """Tag -> category index and rating level, from the group JSONs."""

    def __init__(self, directory=_DIR):
        def load(name):
            with open(os.path.join(directory, name), encoding="utf-8") as f:
                return json.load(f)

        tree = load("hierarchy_v1.0.json")
        cats = load("categories_v1.0.json")
        self._groups = load("tags_v1.0.json")
        self._ratings = load("ratings_v1.0.json")

        self.names = list(cats["priority"])
        self._fallback = len(self.names) - 1        # "etc"

        children = {}

        def walk(node):
            for name, sub in node.items():
                children[name] = sub
                walk(sub)

        walk(tree)

        def descendants(name):
            out, stack = {name}, [children.get(name, {})]
            while stack:
                for key, sub in stack.pop().items():
                    out.add(key)
                    stack.append(sub)
            return out

        # group -> highest-priority category that owns it
        self._group_cat = {}
        for rank, name in enumerate(self.names):
            for root in cats["categories"].get(name, []):
                for group in descendants(root):
                    if group not in self._group_cat:
                        self._group_cat[group] = rank

    def category_of(self, tag):
        """The category most of the tag's groups point at.

        Majority rather than strict priority, because wiki membership is
        noisy in a way priority amplifies: "beach" is filed under
        locations, water *and* swimsuit, and "building" under locations
        and the gerund list, so whichever category ranks highest would
        win on a single stray group. Ties fall back to priority order.
        """
        votes = {}
        for group in self._groups.get(normalize(tag), ()):
            rank = self._group_cat.get(group)
            if rank is not None:
                votes[rank] = votes.get(rank, 0) + 1
        if not votes:
            return self._fallback
        best = max(votes.values())
        return min(rank for rank, n in votes.items() if n == best)

    def rating_of(self, tag, default=3):
        """Rating level index, `default` when the tag has no label.

        Callers that mask by rating want the cautious default (explicit,
        so an unlabelled tag cannot slip into a mild request); callers
        that only describe a tag want the permissive one.
        """
        level = self._ratings.get(normalize(tag))
        return RATING_ORDER.index(level) if level in RATING_ORDER else default

    def category_name(self, tag):
        return self.names[self.category_of(tag)]

    def knows(self, tag):
        return normalize(tag) in self._groups

    def arrays(self, vocab):
        """(category index, rating level) arrays aligned to `vocab`."""
        import numpy as np
        cats = np.fromiter((self.category_of(t) for t in vocab),
                           dtype=np.int8, count=len(vocab))
        levels = np.fromiter((self.rating_of(t) for t in vocab),
                             dtype=np.int8, count=len(vocab))
        return cats, levels


def load_labels():
    global _LABELS
    if _LABELS is None:
        try:
            _LABELS = Labels()
        except Exception:
            _LABELS = False
    return _LABELS


def parse_categories(spec, names):
    """Parse a category request into (allowed ranks, quota by rank).

    "" or None      -> (None, None): every category allowed, no quota
    "pose, clothes" -> only those categories, no per-category limit
    "pose:3, ..."   -> same, and at most 3 tags may come from pose

    Unknown names are ignored; a spec that names nothing usable behaves
    like an empty spec so a typo cannot silence the generator.
    """
    if not spec or not spec.strip():
        return None, None
    rank_of = {name: i for i, name in enumerate(names)}
    allowed, quota = set(), {}
    for item in spec.replace("\n", ",").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, count = item.partition(":")
        rank = rank_of.get(name.strip().lower())
        if rank is None:
            continue
        allowed.add(rank)
        count = count.strip()
        if count.isdigit():
            quota[rank] = quota.get(rank, 0) + int(count)
    if not allowed:
        return None, None
    return allowed, (quota or None)
