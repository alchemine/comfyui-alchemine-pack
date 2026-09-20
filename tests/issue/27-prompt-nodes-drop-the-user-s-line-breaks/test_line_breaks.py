import pytest

# Nothing in here needs filtering, so the only right answer is the input.
UNTOUCHED = (
    "1girl, smile,\nlong hair,  (blue eyes:1.2),\n\nsky BREAK\nred dress, arm up"
)


@pytest.fixture
def prompt(pack):
    return pack("prompt")


@pytest.mark.parametrize(
    "node",
    [
        "FilterTags",
        "FilterSubtags",
        "FilterColors",
        "FilterPlurals",
        "BoySubjectFilter",
    ],
)
def test_a_prompt_with_nothing_to_filter_comes_back_as_it_was(prompt, node):
    assert getattr(prompt, node).execute(text=UNTOUCHED)[0] == UNTOUCHED


def test_process_tags_with_nothing_to_filter_comes_back_as_it_was(prompt):
    assert prompt.ProcessTags.execute(text=UNTOUCHED)[0] == UNTOUCHED


@pytest.mark.parametrize(
    "node, text, kwargs, expected",
    [
        ("FilterTags", "1girl, smile,\nbad, sky,\ntree", {"blacklist_tags": "^bad$"}, "1girl, smile,\nsky,\ntree"),
        ("FilterSubtags", "1girl, smile,\ndog, sky,\nwhite dog", {}, "1girl, smile,\nsky,\nwhite dog"),
        ("FilterColors", "red dress, smile,\nblue dress, sky,\ntree", {}, "red dress, smile,\nsky,\ntree"),
        ("FilterPlurals", "arm up, smile,\narms up, sky,\ntree", {}, "arm up, smile,\nsky,\ntree"),
    ],
)  # fmt: skip
def test_the_line_break_outlives_the_first_tag_of_its_line(
    prompt, node, text, kwargs, expected
):
    assert getattr(prompt, node).execute(text=text, **kwargs)[0] == expected


def test_a_tag_dropped_from_the_middle_of_a_line_takes_only_itself(prompt):
    out = prompt.FilterColors.execute(text="red dress,\nsmile, blue dress, sky")[0]

    assert out == "red dress,\nsmile, sky"


def test_the_last_tag_of_a_prompt_leaves_no_trailing_comma(prompt):
    assert (
        prompt.FilterColors.execute(text="red dress, sky,\nblue dress")[0]
        == "red dress, sky"
    )


def test_remove_weights_keeps_the_layout(prompt):
    out = prompt.RemoveWeights.execute(
        text="(1girl:1.2), (smile),\n[long hair],  sky BREAK\n(red dress:0.9)"
    )[0]

    assert out == "1girl, smile,\nlong hair,  sky BREAK\nred dress"


def test_boy_subject_filter_keeps_the_layout(prompt):
    out = prompt.BoySubjectFilter.execute(
        text="1girl, solo,\nsex, smile", add_tags="hetero"
    )[0]

    assert out == "1girl,\nsex, smile, ((1boy)), hetero"


def test_boy_subject_filter_reads_across_break(prompt):
    out = prompt.BoySubjectFilter.execute(
        text="1girl, solo BREAK sex, smile", add_tags="hetero"
    )[0]

    assert out == "1girl BREAK sex, smile, ((1boy)), hetero"
