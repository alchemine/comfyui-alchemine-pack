# -*- coding: utf-8 -*-
"""Tests for tag_guard. Run: python test_tag_guard.py"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "nodes", "lib"))
from tag_guard import build_ban_tags, filter_generated, classify, normalize


def as_set(s):
    return set(t.strip() for t in s.split(",") if t.strip())


def test_bikini_bans_other_outfits():
    ban, report = build_ban_tags("1girl, bikini, beach")
    b = as_set(ban)
    for t in ["dress", "shirt", "pants", "skirt", "white dress", "t-shirt",
              "school uniform", "panties", "sundress", "one-piece swimsuit"]:
        assert t in b, "%s should be banned" % t
    # not banned: the tag itself, non-conflicting categories
    assert "bikini" not in b
    assert "thighhighs" not in b, "legwear does not conflict with bikini"
    assert "jacket" not in b, "outerwear allowed unless strict"
    assert "standing" not in b, "pose untouched"
    # background 'beach' detected -> other locations banned
    assert "classroom" in b
    assert "beach" not in b
    print("test_bikini_bans_other_outfits OK")


def test_strict_bans_outerwear():
    ban, _ = build_ban_tags("bikini", clothes_strict=True)
    b = as_set(ban)
    assert "jacket" in b and "coat" in b
    print("test_strict_bans_outerwear OK")


def test_top_bottom_coexist():
    ban, _ = build_ban_tags("white shirt, black skirt")
    b = as_set(ban)
    assert "dress" in b, "full outfit conflicts with top"
    assert "t-shirt" in b, "second top banned"
    assert "pleated skirt" in b, "second bottom banned"
    assert "thighhighs" not in b
    print("test_top_bottom_coexist OK")


def test_hair_and_eyes():
    ban, _ = build_ban_tags("blonde hair, blue eyes, twintails")
    b = as_set(ban)
    assert "black hair" in b and "red eyes" in b and "ponytail" in b
    assert "long hair" not in b, "hair length is a separate category"
    print("test_hair_and_eyes OK")


def test_ban_all_for_category_scoping():
    # problem #2 preview: lock everything except pose
    modes = {c: "ban_all" for c in
             ("clothes", "expression", "hair_length", "hair_style",
              "hair_color", "eye_color", "background")}
    modes["pose"] = "off"
    ban, _ = build_ban_tags("1girl", modes=modes)
    b = as_set(ban)
    assert "dress" in b and "smile" in b and "classroom" in b
    assert "standing" not in b and "sitting" not in b
    print("test_ban_all_for_category_scoping OK")


def test_underscore_io():
    ban, _ = build_ban_tags("blonde_hair", use_underscores=True)
    b = as_set(ban)
    assert "black_hair" in b
    assert "blonde_hair" not in b
    print("test_underscore_io OK")


def test_filter_catches_variants():
    filtered, removed = filter_generated(
        "1girl, solo, white frilled dress, standing, black thighhighs, smile",
        locked_prompt="bikini")
    r = as_set(removed)
    f = as_set(filtered)
    assert "white frilled dress" in r, "pattern catches unseen dress variant"
    assert "black thighhighs" in f and "smile" in f and "standing" in f
    print("test_filter_catches_variants OK")


def test_filter_self_consistency():
    filtered, removed = filter_generated(
        "1girl, ponytail, twintails, blonde hair, black hair")
    r = as_set(removed)
    f = as_set(filtered)
    assert "twintails" in r, "second hair style dropped"
    assert "black hair" in r, "second hair color dropped"
    assert "ponytail" in f and "blonde hair" in f
    print("test_filter_self_consistency OK")


def test_classify_and_normalize():
    assert classify("micro bikini") == ("clothes", "full")
    assert classify(normalize("Blonde_Hair")) == ("hair_color", None)
    assert classify("purple sundress") == ("clothes", "full")  # pattern
    assert classify("closed eyes") != ("eye_color", None)
    assert classify("1girl") == (None, None)
    print("test_classify_and_normalize OK")


def test_weighted_tags():
    ban, _ = build_ban_tags("(bikini:1.2), beach")
    assert "dress" in as_set(ban)
    print("test_weighted_tags OK")


def test_subject_tags():
    from tag_guard import is_subject, filter_by_conflicts
    for t in ("1boy", "2girls", "6+boys", "solo", "male focus", "no humans"):
        assert is_subject(t), t
    for t in ("large breasts", "1girl hugging own leg", "boy"):
        assert not is_subject(t), t
    # subject tags are never removed by the cooc rule (1boy vs large breasts
    # has cos 0.82 / lift 0.00 but is not a substitution conflict)
    kept, report = filter_by_conflicts("1boy, smile", "large breasts")
    assert "1boy" in kept and "REMOVED" not in report, (kept, report)
    print("test_subject_tags OK")


def test_classify_tags_buckets():
    from tag_classify import classify_tags
    r = classify_tags("1boy, serafuku, sitting, smile, classroom, "
                      "silver hair, tennis racket, qqzzxx")
    assert r["characters"] == ["1boy"]
    assert r["clothes"] == ["serafuku"]
    assert r["pose"] == ["sitting"]
    assert r["expression"] == ["smile"]
    assert r["background"] == ["classroom"]
    assert r["body"] == ["silver hair"]
    assert r["objects"] == ["tennis racket"]
    assert r["others"] == ["qqzzxx"]
    print("test_classify_tags_buckets OK")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nAll tests passed.")
