# -*- coding: utf-8 -*-
"""Classify prompt tags into coarse buckets using the Danbooru tag-group
mapping (tag_groups.json, built by docs/tag-consistency-guard/
extract_tag_groups.py) with the static tag_data tables as fallback.

Standalone (no ComfyUI dependency) so it can be unit-tested and reused.
"""
import json
import os

try:
    from .tag_guard import classify, is_subject, split_prompt
except ImportError:
    from tag_guard import classify, is_subject, split_prompt

BUCKETS = ("characters", "clothes", "body", "expression", "pose",
           "background", "objects", "nsfw", "others")

# wiki tag group -> bucket. Groups not listed fall through to "others".
_GROUP_BUCKET = {
    "people": "characters",
    "groups": "characters",
    "family relationships": "characters",
    "jobs": "characters",

    "attire": "clothes",
    "dress": "clothes",
    "headwear": "clothes",
    "sleeves": "clothes",
    "neck and neckwear": "clothes",
    "eyewear": "clothes",
    "legwear": "clothes",
    "panties": "clothes",
    "bra": "clothes",
    "sexual attire": "clothes",
    "fashion style": "clothes",
    "prints": "clothes",
    "patterns": "clothes",
    "mask": "clothes",
    "embellishment": "clothes",
    "covering": "clothes",

    "body parts": "body",
    "breasts tags": "body",
    "hands": "body",
    "shoulders": "body",
    "ass": "body",
    "tail": "body",
    "wings": "body",
    "ears tags": "body",
    "skin color": "body",
    "hair": "body",
    "hair styles": "body",
    "hair color": "body",
    "piercings": "body",

    "face tags": "expression",
    "eyes tags": "expression",

    "posture": "pose",
    "verbs and gerunds": "pose",
    "gestures": "pose",
    "dances": "pose",
    "sexual positions": "pose",

    "locations": "background",
    "backgrounds": "background",
    "real world locations": "background",
    "water": "background",
    "fire": "background",
    "lighting": "background",

    "food tags": "objects",
    "technology": "objects",
    "symbols": "objects",
    "cards": "objects",
    "board games": "objects",
    "video game": "objects",
    "sports": "objects",
    "flowers": "objects",
    "cats": "objects",
    "dogs": "objects",
    "birds": "objects",
    "animals": "objects",
    "legendary creatures": "objects",
    "sex objects": "objects",
    "audio tags": "objects",

    "sex acts": "nsfw",
    "simulated sex acts": "nsfw",
    "nudity": "nsfw",
    "pussy": "nsfw",
    "censorship": "nsfw",
}

# a tag can belong to several groups -> several buckets; pick by priority
_BUCKET_PRIORITY = ("characters", "nsfw", "clothes", "pose", "expression",
                    "body", "background", "objects", "others")

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

_GROUPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "resources", "tag_groups.json")
_GROUPS = None  # lazy: dict or False if unavailable


def _load_groups():
    global _GROUPS
    if _GROUPS is None:
        try:
            with open(_GROUPS_PATH) as f:
                _GROUPS = json.load(f)
        except Exception:
            _GROUPS = False
    return _GROUPS


def bucket_of(tag):
    """Return the bucket name for a normalized tag."""
    if is_subject(tag):
        return "characters"
    groups = _load_groups() or {}
    buckets = {_GROUP_BUCKET.get(g, "others") for g in groups.get(tag, ())}
    if buckets:
        for b in _BUCKET_PRIORITY:
            if b in buckets:
                return b
    cat, _sub = classify(tag)
    if cat is not None:
        return _STATIC_BUCKET.get(cat, "others")
    # last-resort suffix heuristics for common unmapped variants
    # (e.g. 'silver hair' is in neither the wiki groups nor tag_data)
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
