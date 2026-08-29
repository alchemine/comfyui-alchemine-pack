# -*- coding: utf-8 -*-
"""Classify prompt tags into coarse buckets.

Buckets come from the same labels TagGenerator samples with
(resources/group/, via tag_category): the tag's category decides the
bucket, and its rating level decides whether "nsfw" overrides that.
Tags the labels do not cover fall back to the static tag_data tables.

The bucket names predate those categories and are kept as they are so
existing workflows keep their wiring; `compositions` is appended rather
than inserted for the same reason.

Standalone (no ComfyUI dependency) so it can be unit-tested and reused.
"""
try:
    from . import tag_category
    from .tag_guard import classify, is_subject, split_prompt
except ImportError:
    import tag_category
    from tag_guard import classify, is_subject, split_prompt

BUCKETS = ("characters", "clothes", "body", "expression", "pose",
           "background", "objects", "nsfw", "others", "compositions")

# category (categories_v1.0.json) -> bucket. creatures folds into
# objects, which is where the old mapping put cats, dogs and elves too.
_CATEGORY_BUCKET = {
    "characters": "characters",
    "expressions": "expression",
    "pose": "pose",
    "clothes": "clothes",
    "background": "background",
    "compositions": "compositions",
    "body": "body",
    "objects": "objects",
    "creatures": "objects",
    "etc": "others",
}

# rating level at or above which a tag is bucketed nsfw regardless of
# category: q and e. Level s (swimsuit, cleavage) stays with its
# category, as it did when nsfw was decided by wiki group.
_NSFW_LEVEL = 2

# tag_data classify() category -> bucket, for tags without a wiki group
_STATIC_BUCKET = {
    "clothes": "clothes",
    "accessories": "clothes",
    "pose": "pose",
    "expression": "expression",
    "eye_color": "expression",
    "hair_length": "body",
    "hair_style": "body",
    "hair_color": "body",
    "background": "background",
}

def bucket_of(tag):
    """Return the bucket name for a normalized tag."""
    if is_subject(tag):
        return "characters"

    labels = tag_category.load_labels()
    if labels:
        # unlabelled tags must not read as explicit here: this is a
        # description, not a filter, so an unknown tag falls through
        # rather than being called nsfw
        if labels.rating_of(tag, default=0) >= _NSFW_LEVEL:
            return "nsfw"
        if labels.knows(tag):
            bucket = _CATEGORY_BUCKET.get(labels.category_name(tag))
            if bucket and bucket != "others":
                return bucket

    cat, _sub = classify(tag)
    if cat is not None:
        return _STATIC_BUCKET.get(cat, "others")
    # last-resort suffix heuristics for common unmapped variants
    if tag.endswith(" hair"):
        return "body"
    if tag.endswith(" eyes"):
        return "expression"
    return "others"


def classify_tags(text):
    """Split a prompt and group its tags by bucket.

    Returns {bucket: [tag, ...]} with every bucket present (possibly
    empty), tags in input order, duplicates dropped.
    """
    out = {b: [] for b in BUCKETS}
    seen = set()
    for t in split_prompt(text):
        if t in seen:
            continue
        seen.add(t)
        out[bucket_of(t)].append(t)
    return out
