import pytest


@pytest.fixture
def prompt(pack):
    return pack("prompt")


@pytest.mark.parametrize(
    "tag, bare",
    [
        ("cat", "cat"),
        ("(cat)", "cat"),
        ("((cat))", "cat"),
        ("(((cat)))", "cat"),
        ("[cat]", "cat"),
        ("[[cat]]", "cat"),
        ("(cat:1.2)", "cat"),
        # a literal paren is part of the tag, however many brackets wrap it
        (r"star \(sky\)", r"star \(sky\)"),
        (r"(star \(sky\))", r"star \(sky\)"),
        (r"((star \(sky\)))", r"star \(sky\)"),
        # not emphasis at all
        (":)", ":)"),
        ("(cat", "(cat"),
    ],
)
def test_the_tag_comes_out_of_its_brackets_whole(prompt, tag, bare):
    assert prompt.BasePrompt.remove_weight(tag) == bare
    assert prompt.BasePrompt.normalize_tag(tag) == bare


def test_remove_weights_on_nested_emphasis(prompt):
    assert (
        prompt.RemoveWeights.execute(text="((cat)), [[dog]], (bird)")[0]
        == "cat, dog, bird"
    )
