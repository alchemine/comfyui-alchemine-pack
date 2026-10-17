import pytest


@pytest.fixture
def run(pack):
    def _run(captions, threshold=0.8, fixed_tags=""):
        return pack("lora").FilterFrequentTags.execute(
            captions=captions, threshold=[threshold], fixed_tags=[fixed_tags]
        )
    return _run


CAPS = ["a, b, c", "a, b, d", "a, b, e", "a, c, f", "a, x, y"]


def test_tags_at_or_above_the_threshold_go(run):
    caps, removed = run(CAPS, threshold=0.8)
    assert caps == ["b, c", "b, d", "b, e", "c, f", "x, y"]
    assert removed == "a(5/5)"


def test_tags_below_the_threshold_stay(run):
    caps, removed = run(CAPS, threshold=0.6)
    assert caps == ["c", "d", "e", "c, f", "x, y"]
    assert removed == "a(5/5), b(3/5)"


def test_fixed_tags_are_kept(run):
    caps, removed = run(CAPS, threshold=0.8, fixed_tags="a")
    assert caps == CAPS
    assert removed == ""


def test_order_is_preserved(run):
    caps, _ = run(["z, a, y", "a, z", "q, a"], threshold=1.0)
    assert caps == ["z, y", "z", "q"]


def test_whitespace_is_normalized(run):
    caps, removed = run(["a ,  b", " a, c ", "a,d"], threshold=1.0)
    assert caps == ["b", "c", "d"]
    assert removed == "a(3/3)"


def test_empty_captions_pass_through(run):
    caps, removed = run(["", "a, b", ""], threshold=0.3)
    assert caps == ["", "", ""]
    assert removed == "a(1/3), b(1/3)"


def test_a_single_caption_loses_every_tag_at_threshold_one(run):
    caps, removed = run(["a, b"], threshold=1.0, fixed_tags="a")
    assert caps == ["a"]
    assert removed == "b(1/1)"


def test_a_tag_repeated_inside_one_caption_counts_once(run):
    caps, removed = run(["a, a, b", "c, b"], threshold=1.0)
    assert caps == ["a, a", "c"]
    assert removed == "b(2/2)"
