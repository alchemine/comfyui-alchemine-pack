import pytest

ADDED = "((1boy)), (hetero:1.1), (couple:1.1), (deep skin:1.1)"


@pytest.fixture
def run(pack):
    def run(text, **kwargs):
        return pack("prompt").BoySubjectFilter.execute(text=text, **kwargs)[0]

    return run


def test_no_boy_counts_him_in_and_takes_solo_out(run):
    assert run("1girl, solo, sex, smile") == f"1girl, sex, smile, {ADDED}"


def test_a_counted_boy_only_loses_solo(run):
    assert run("1girl, 1boy, solo, sex") == "1girl, 1boy, sex"


def test_add_tags_is_what_comes_in_with_him(run):
    assert run("1girl, sex", add_tags="hetero") == "1girl, sex, ((1boy)), hetero"


@pytest.mark.parametrize(
    "text",
    [
        "1girl, solo, smile",  # nothing needs a man
        "1girl, 1boy, solo",  # a counted boy alone is not a reason
        "1girl, solo, sex toy, sexy, unisex",  # whole tags, not substrings
        "1girl, solo, after sex, after fellatio",  # the aftermath holds alone
    ],
)
def test_left_alone(run, text):
    assert run(text) == text


@pytest.mark.parametrize(
    "tag",
    [
        "grabbing another's hand",  # spelled with "another"
        "double penetration",  # a pattern, not a listed tag
        "reverse cowgirl position",
        "erection",  # a male body
        "Sex_From_Behind",  # underscores and case do not hide a tag
    ],
)
def test_detected(run, tag):
    assert run(f"1girl, solo, {tag}") == f"1girl, {tag}, {ADDED}"


def test_a_male_body_counts_even_next_to_futanari(run):
    assert (
        run("1girl, solo, futanari, erection") == f"1girl, futanari, erection, {ADDED}"
    )


def test_multiple_boys_is_a_counted_boy(run):
    assert (
        run("1girl, multiple boys, solo, gangbang") == "1girl, multiple boys, gangbang"
    )
