import pytest


@pytest.fixture
def prompt(pack):
    return pack("prompt")


@pytest.mark.parametrize(
    "text, pieces",
    [
        # emphasis keeps its commas, as before
        ("(masterpiece), (best quality:1.2), (highres, absurdres)", ["(masterpiece)", " (best quality:1.2)", " (highres, absurdres)"]),
        ("((a, b), c:1.1), d", ["((a, b), c:1.1)", " d"]),
        # a paren with no partner is a literal: the "(" of ">:(", the ")" of ":)"
        (">:(, (smile:1.2), sky", [">:(", " (smile:1.2)", " sky"]),
        (":), (smile:1.2), sky", [":)", " (smile:1.2)", " sky"]),
        ("sky, (smile, :), tree", ["sky", " (smile, :)", " tree"]),
        # an escaped paren is a literal too, inside emphasis or out
        (r"star \(sky\), (smile:1.2)", [r"star \(sky\)", " (smile:1.2)"]),
        (r"(star \(sky\), night:1.2), moon", [r"(star \(sky\), night:1.2)", " moon"]),
        (r"pozyomka \(arknights, moon", [r"pozyomka \(arknights", " moon"]),
    ],
)  # fmt: skip
def test_split_tags(prompt, text, pieces):
    assert prompt.BasePrompt.split_tags(text) == pieces


def test_remove_weights_after_an_emoticon(prompt):
    run = lambda text: prompt.RemoveWeights.execute(text=text)[0]

    assert run(">:(, (smile:1.2),\n[sky]") == ">:(, smile,\nsky"
    assert run(":), (smile:1.2),\n[sky]") == ":), smile,\nsky"


def test_an_opener_and_a_closer_pair_up_even_when_both_are_emoticons(prompt):
    # ">:(" before ":)" is, to the prompt syntax, one emphasis group -- ComfyUI
    # reads it that way too. Escaping either one is how to say otherwise.
    assert prompt.BasePrompt.split_tags(">:(, smile, :), sky") == [
        ">:(, smile, :)",
        " sky",
    ]
    assert prompt.BasePrompt.split_tags(r">:\(, smile, :\), sky") == [
        r">:\(",
        " smile",
        r" :\)",
        " sky",
    ]
