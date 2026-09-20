import pytest

TEXT = "1girl, red_dress, blue dress, arm up, arms up, dog, white dog"


@pytest.fixture
def run(pack):
    return lambda **kwargs: pack("prompt").ProcessTags.execute(text=TEXT, **kwargs)


def test_both_steps_run_by_default_after_the_subtags(run):
    text, filtered = run()

    assert text == "1girl, red dress, arm up, white dog"
    assert filtered == ["dog", "blue dress", "arms up"]


def test_each_step_has_its_own_switch(run):
    assert (
        run(filter_colors=False)[0] == "1girl, red dress, blue dress, arm up, white dog"
    )
    assert (
        run(filter_plurals=False)[0] == "1girl, red dress, arm up, arms up, white dog"
    )


def test_the_widgets_sit_under_filter_subtags(pack):
    required = list(pack("prompt").ProcessTags.INPUT_TYPES()["required"])

    at = required.index("filter_subtags")
    assert required[at + 1 : at + 4] == [
        "filter_colors",
        "filter_plurals",
        "auto_break",
    ]


def test_break_survives_the_new_steps(pack):
    text, _ = pack("prompt").ProcessTags.execute(
        text="red dress, smile BREAK blue dress, sky"
    )

    assert text == "red dress, smile BREAK sky"
