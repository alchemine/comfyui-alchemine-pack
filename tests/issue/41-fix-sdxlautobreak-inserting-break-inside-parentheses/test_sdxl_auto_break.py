import re
import types

import pytest

START, END, PAD = 1, 2, 0


class FakeClip:
    """One token per word or punctuation mark, as CLIP splits them.

    Like ComfyUI, only the first 77 token chunk is returned here, which is all
    SDXLAutoBreak reads.
    """

    tokenizer = types.SimpleNamespace(
        clip_g=types.SimpleNamespace(start_token=START, end_token=END, pad_token=PAD)
    )

    def tokenize(self, text):
        words = re.findall(r"\w+|[^\w\s]", text)
        chunk = [(START, 1.0)] + [(10, 1.0)] * min(len(words), 75) + [(END, 1.0)]
        return {"g": [chunk]}


@pytest.fixture
def run(pack):
    return lambda text: pack("prompt").SDXLAutoBreak.execute(clip=FakeClip(), text=text)[0]


TAIL = "(white background,:-1) (@ @,:-1.1) (light particles,:-1.2) posing, shade"


@pytest.mark.parametrize("n", range(20, 32))
def test_break_stays_out_of_parentheses(run, n):
    out = run(", ".join(["red apple"] * n) + ", " + TAIL)

    assert "BREAK" in out
    for segment in out.split("BREAK"):
        assert segment.count("(") == segment.count(")"), out


def test_break_still_splits_between_tags(run):
    out = run(", ".join(["red apple"] * 30))

    assert out.count("BREAK") == 1
    assert out.replace("\n\nBREAK\n", ", ") == ", ".join(["red apple"] * 30)
