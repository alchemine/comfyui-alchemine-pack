import io
import json
import time
import zlib

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def image(pack):
    return pack("save_image")


@pytest.fixture
def save(image, comfy_dirs):
    def run(pixels=None, prefix="save", **kw):
        pixels = torch.full((1, 8, 8, 3), 0.5) if pixels is None else pixels
        files = image.FastImageSave.execute(pixels, prefix, **kw)["ui"]["images"]
        return [comfy_dirs.output / f["subfolder"] / f["filename"] for f in files]

    return run


def read_segments(path):
    data = path.read_bytes()
    offset, chunks, markers = 2, [], []
    while data[offset + 1] != 0xDA:
        length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        markers.append(data[offset + 1])
        if data[offset + 1] == 0xEF:
            assert data[offset + 4 : offset + 12] == b"ComfyUI\0"
            chunks.append(data[offset + 12 : offset + 2 + length])
        offset += 2 + length
    return markers, chunks


def read_metadata(path):
    strings = json.loads(zlib.decompress(b"".join(read_segments(path)[1])))
    return {key: json.loads(value) for key, value in strings.items()}


def test_workflow_and_prompt_come_back_unchanged(save):
    workflow = {"nodes": [{"id": 1, "title": "한글"}]}
    prompt = {"1": {"class_type": "KSampler", "inputs": {}}}
    path = save(prompt=prompt, extra_pnginfo={"workflow": workflow})[0]
    assert read_metadata(path) == {"workflow": workflow, "prompt": prompt}


def test_nan_in_the_prompt_leaves_the_workflow_strict_json(save):
    path = save(
        prompt={"1": {"is_changed": [float("nan")]}},
        extra_pnginfo={"workflow": {"a": 1}},
    )[0]
    strings = json.loads(zlib.decompress(b"".join(read_segments(path)[1])))

    def reject(constant):
        raise ValueError(constant)

    assert json.loads(strings["workflow"], parse_constant=reject) == {"a": 1}
    with pytest.raises(ValueError):
        json.loads(strings["prompt"], parse_constant=reject)


def test_other_extra_pnginfo_keys_are_kept(save):
    path = save(extra_pnginfo={"workflow": {}, "other": [1]})[0]
    assert read_metadata(path) == {"workflow": {}, "other": [1]}


def test_metadata_over_one_segment_is_split(save):
    noise = np.random.default_rng(0).integers(0, 2**32, 40_000).tolist()
    path = save(extra_pnginfo={"workflow": {"noise": noise}})[0]
    assert len(read_segments(path)[1]) > 1
    assert read_metadata(path)["workflow"]["noise"] == noise


def test_no_metadata_writes_no_segment(save):
    assert read_segments(save()[0])[1] == []


def test_disable_metadata_writes_no_segment(save, image, monkeypatch):
    monkeypatch.setattr(image.args, "disable_metadata", True)
    assert read_segments(save(prompt={"1": {}})[0])[1] == []


def test_jfif_stays_the_first_segment(save):
    assert read_segments(save(prompt={"1": {}})[0])[0][:2] == [0xE0, 0xEF]


def test_pil_decodes_the_same_pixels_with_and_without_metadata(save):
    pixels = torch.linspace(0, 1, 64 * 3).reshape(1, 8, 8, 3)
    plain, embedded = save(pixels)[0], save(pixels, prompt={"1": {}})[0]
    assert np.array_equal(
        np.asarray(Image.open(plain)), np.asarray(Image.open(embedded))
    )


def test_out_of_range_pixels_are_clamped(save):
    path = save(torch.tensor([[[[-1.0, 2.0, 0.5]]]]))[0]
    r, g, _ = Image.open(io.BytesIO(path.read_bytes())).getpixel((0, 0))
    assert r < 40 and g > 215


def test_batch_gets_one_file_per_image_each_with_metadata(save):
    paths = save(torch.zeros(3, 8, 8, 3), prefix="batch", prompt={"1": {}})
    assert [p.name for p in paths] == [
        "batch_00001.jpg",
        "batch_00002.jpg",
        "batch_00003.jpg",
    ]
    assert all(read_metadata(p) == {"prompt": {"1": {}}} for p in paths)


def test_counter_continues_after_existing_files(save):
    save(prefix="cont")
    assert save(prefix="cont")[0].name == "cont_00002.jpg"


def test_counter_finds_the_end_of_a_long_run(save, comfy_dirs):
    for number in range(1, 1668):
        (comfy_dirs.output / f"long_{number:05}.jpg").touch()
    assert save(prefix="long")[0].name == "long_01668.jpg"


def test_a_gap_never_overwrites_the_files_after_it(save, comfy_dirs):
    for number in (1, 3):
        (comfy_dirs.output / f"gap_{number:05}.jpg").write_bytes(b"keep")
    paths = save(torch.zeros(2, 8, 8, 3), prefix="gap")
    assert [p.name for p in paths] == ["gap_00002.jpg", "gap_00004.jpg"]
    assert (comfy_dirs.output / "gap_00003.jpg").read_bytes() == b"keep"


def test_files_of_another_stem_do_not_count(save, comfy_dirs):
    (comfy_dirs.output / "day-00_00001.jpg").touch()
    assert save(prefix="day")[0].name == "day_00001.jpg"


def test_prefix_makes_subfolders_and_expands_variables(save):
    path = save(torch.zeros(1, 2, 3, 3), prefix="%year%/%width%x%height%")[0]
    assert path.parent.name == time.strftime("%Y") and path.name == "3x2_00001.jpg"


def test_prefix_cannot_leave_the_output_folder(save, comfy_dirs):
    with pytest.raises(Exception):
        save(prefix="../escape")
    assert not (comfy_dirs.root / "escape_00001.jpg").exists()


def test_images_are_output_unchanged(image):
    pixels = torch.zeros(2, 8, 8, 3)
    assert image.FastImageSave.execute(pixels, "out")["result"][0] is pixels
    assert image.FastImageSave.RETURN_NAMES == ("images",)
