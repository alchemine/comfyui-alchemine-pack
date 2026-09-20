import pytest


@pytest.fixture
def run(pack):
    return lambda text: pack("prompt").FilterPlurals.execute(text=text)


def test_the_later_spelling_goes(run):
    assert run("1girl, arm up, arms up, smile") == ("1girl, arm up, smile", "arms up")
    assert run("1girl, arms up, arm up, smile") == ("1girl, arms up, smile", "arm up")


def test_any_word_of_the_tag_may_carry_the_s(run):
    assert run("hand on hip, hands on hips, boots, boot") == (
        "hand on hip, boots",
        "hands on hips, boot",
    )


def test_short_words_are_not_plurals(run):
    text = "ass, as, abs, ab"
    assert run(text) == (text, "")


def test_a_different_tag_is_not_a_plural(run):
    text = "glass, glasses, dress, dresses, arm up, arm down"
    assert run(text) == (text, "")


def test_the_same_tag_twice_is_not_this_filters_business(run):
    assert run("arms up, arms up") == ("arms up, arms up", "")


def test_weights_and_break(run):
    assert run("(arm up:1.2), smile BREAK arms up, sky") == (
        "(arm up:1.2), smile BREAK sky",
        "arms up",
    )
