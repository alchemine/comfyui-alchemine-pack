import pytest


@pytest.fixture
def run(pack):
    return lambda text: pack("prompt").SeparateLoraTags.execute(text=text)


@pytest.mark.parametrize(
    "text, without",
    [
        # the lora is the first thing on its line
        ("1girl,\n<lora:a:0.8>, smile,\n<lora:b:1>", "1girl,\nsmile"),
        ("1girl,\n<lora:a:0.8> blonde, sky", "1girl,\nblonde, sky"),
        ("1girl,\n\n<lora:a:0.8>, smile", "1girl,\n\nsmile"),
        # a line of nothing but loras goes, and takes its own line break with it
        ("1girl,\n<lora:a:1>,\nsky", "1girl,\nsky"),
        ("1girl,\n<lora:a:1> <lora:b:1>\nsky", "1girl,\nsky"),
        # the lora is further along its line
        ("1girl, <lora:a:1>, smile,\nsky", "1girl, smile,\nsky"),
        ("1girl, smile <lora:a:1>,\nsky", "1girl, smile,\nsky"),
        ("1girl, smile, <lora:a:1>\nsky", "1girl, smile,\nsky"),
    ],
)
def test_the_lines_stay_where_they_were(run, text, without):
    assert run(text)[0] == without


@pytest.mark.parametrize(
    "text, without",
    [
        ("1girl, <lora:a:1>, smile", "1girl, smile"),
        ("<lora:a:1>, 1girl", "1girl"),
        ("1girl, smile <lora:a:1>", "1girl, smile"),
        ("1girl, <lora:a:1> blonde, sky", "1girl, blonde, sky"),
        ("1girl,\n\nsky", "1girl,\n\nsky"),
    ],
)
def test_one_line_prompts_come_out_as_before(run, text, without):
    assert run(text)[0] == without


def test_the_example_in_the_docstring(run):
    text = (
        "1girl, <lora:a.safetensors:0.7> blonde, jewelry,\n"
        "<lora:b.safetensors:0.7> <lora:c.safetensors:0.7> <lora:c.safetensors:1.0>"
    )

    assert run(text) == (
        "1girl, blonde, jewelry",
        "<lora:a.safetensors:0.7> <lora:b.safetensors:0.7> <lora:c.safetensors:1.0>",
    )
