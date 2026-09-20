import pytest


@pytest.fixture
def run(pack):
    return lambda text: pack("prompt").FilterColors.execute(text=text)


def test_the_first_colour_of_a_thing_stays(run):
    assert run("1girl, red dress, blue dress, white shirt, black shirt") == (
        "1girl, red dress, white shirt",
        "blue dress, black shirt",
    )


def test_different_things_keep_their_colours(run):
    text = "black hair, black dress, blue eyes, long hair"
    assert run(text) == (text, "")


def test_a_two_word_colour_is_one_colour(run):
    # "light blue dress" is a dress, not a "blue dress" that is light
    assert run("light blue dress, red dress") == ("light blue dress", "red dress")
    assert run("light blue dress, blue light") == ("light blue dress, blue light", "")


def test_weights_do_not_hide_a_colour(run):
    assert run("(red dress:1.2), (blue dress)") == ("(red dress:1.2)", "(blue dress)")


def test_break_keeps_its_place_and_the_groups_are_one_picture(run):
    assert run("red dress, smile BREAK blue dress, sky") == (
        "red dress, smile BREAK sky",
        "blue dress",
    )
