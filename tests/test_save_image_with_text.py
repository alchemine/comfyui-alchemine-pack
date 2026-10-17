import torch
import pytest


@pytest.fixture
def save(pack, comfy_dirs):
    node = pack("lora").SaveImageWithText
    image = torch.zeros(1, 4, 4, 3)
    return lambda **kw: node.execute(image=image, text=kw.pop("text", "a, b"), dir_path="ds", **kw)


def test_name_sets_the_file_stem(save, comfy_dirs):
    assert save(name="/data/raw/sc_1.jpg") == ("ds/sc_1.png", "ds/sc_1.txt")
    assert (comfy_dirs.output / "ds" / "sc_1.txt").read_text() == "a, b"


def test_an_existing_pair_is_skipped(save, comfy_dirs):
    save(name="sc_2.jpg", text="first")
    assert save(name="sc_2.jpg", text="second") == ("ds/sc_2.png", "ds/sc_2.txt")
    assert (comfy_dirs.output / "ds" / "sc_2.txt").read_text() == "first"


def test_overwrite_rewrites_the_pair(save, comfy_dirs):
    save(name="sc_3.jpg", text="first")
    save(name="sc_3.jpg", text="second", overwrite=True)
    assert (comfy_dirs.output / "ds" / "sc_3.txt").read_text() == "second"


def test_without_name_the_running_number_still_applies(save):
    first = save(prefix="x")
    second = save(prefix="x")
    assert first == ("ds/x_1.png", "ds/x_1.txt") and second == ("ds/x_2.png", "ds/x_2.txt")


def test_only_the_stem_of_name_is_used_so_it_cannot_escape_the_folder(save):
    assert save(name="../../etc/passwd.jpg") == ("ds/passwd.png", "ds/passwd.txt")
