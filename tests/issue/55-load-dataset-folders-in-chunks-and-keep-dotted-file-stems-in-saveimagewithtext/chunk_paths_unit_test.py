import pytest


# (images in the folder, count, chunk, expected names)
CASES = [
    (5, 0, 0, ["01", "02", "03", "04", "05"]),  # count 0 loads everything
    (5, 0, 3, ["01", "02", "03", "04", "05"]),  # chunk is ignored without count
    (5, 2, 0, ["01", "02"]),
    (5, 2, 1, ["03", "04"]),
    (5, 2, 2, ["05"]),  # last group is short
    (5, 2, 3, []),  # past the end
    (5, 5, 0, ["01", "02", "03", "04", "05"]),
    (5, 9, 0, ["01", "02", "03", "04", "05"]),  # count above the folder size
    (5, 1, 4, ["05"]),
    (0, 3, 0, []),  # empty folder
]


@pytest.mark.parametrize("n, count, chunk, expected", CASES)
def test_chunk_paths_slices_the_sorted_images(pack, dataset, n, count, chunk, expected):
    node = pack("lora").LoadImagesFromDir
    paths = node.chunk_paths(dataset(n), count, chunk)
    assert [p.stem for p in paths] == expected
