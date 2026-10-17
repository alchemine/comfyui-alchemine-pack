import pytest
from PIL import Image


@pytest.fixture
def folder(pack, comfy_dirs, request):
    node = pack("lora").LoadImagesFromDir
    d = comfy_dirs.output / request.node.name
    d.mkdir()
    Image.new("RGB", (4, 2), (255, 0, 0)).save(d / "b.jpg")
    Image.new("RGBA", (2, 2), (0, 0, 255, 0)).save(d / "a.png")
    (d / "notes.txt").write_text("skip me")
    return node, d


def test_images_and_names_come_sorted_as_rgb(folder):
    node, d = folder
    images, names = node.execute(d.name)
    assert names == ["a.png", "b.jpg"]
    assert [tuple(i.shape) for i in images] == [(1, 2, 2, 3), (1, 2, 4, 3)]
    assert images[0][0, 0, 0].tolist() == [0.0, 0.0, 1.0]  # alpha dropped


def test_a_missing_folder_yields_empty_lists(pack):
    assert pack("lora").LoadImagesFromDir.execute("nope") == ([], [])


def test_is_changed_follows_the_files(folder):
    node, d = folder
    before = node.IS_CHANGED(d.name)
    Image.new("RGB", (1, 1)).save(d / "c.webp")
    assert node.IS_CHANGED(d.name) != before


def test_the_folder_cannot_escape_output(pack):
    with pytest.raises(ValueError):
        pack("lora").LoadImagesFromDir.execute("../../etc")
