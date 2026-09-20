import io
import types

import pytest
import torch
from PIL import Image


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def lora(pack, monkeypatch):
    module = pack("lora")
    response = types.SimpleNamespace(content=png_bytes(), raise_for_status=lambda: None)
    monkeypatch.setattr(module._session, "get", lambda url: response)
    return module


def test_download_image_writes_inside_output(lora, comfy_dirs):
    _, file_path = lora.DownloadImage.execute("http://host/image.png", "downloads")

    assert file_path == "downloads/1.png"
    assert (comfy_dirs.output / "downloads" / "1.png").is_file()


def test_download_image_dir_path_cannot_leave_output(lora, comfy_dirs):
    with pytest.raises(ValueError):
        lora.DownloadImage.execute("http://host/image.png", "../escaped_download")

    assert not (comfy_dirs.root / "escaped_download").exists()


def test_save_image_with_text_writes_inside_output(lora, comfy_dirs):
    image = torch.zeros(1, 8, 8, 3)

    image_path, text_path = lora.SaveImageWithText.execute(image, "1girl", "saved")

    assert (image_path, text_path) == ("saved/1.png", "saved/1.txt")
    assert (comfy_dirs.output / "saved" / "1.txt").read_text() == "1girl"


def test_save_image_with_text_dir_path_cannot_leave_output(lora, comfy_dirs):
    image = torch.zeros(1, 8, 8, 3)

    with pytest.raises(ValueError):
        lora.SaveImageWithText.execute(image, "1girl", "../escaped_save")

    assert not (comfy_dirs.root / "escaped_save").exists()


def test_save_image_with_text_prefix_cannot_leave_output(lora, comfy_dirs):
    image = torch.zeros(1, 8, 8, 3)
    outside = comfy_dirs.root / "escaped_prefix"
    outside.mkdir()

    with pytest.raises(ValueError):
        lora.SaveImageWithText.execute(
            image, "1girl", "saved", prefix="../../escaped_prefix/p"
        )

    assert list(outside.iterdir()) == []


def test_load_workflow_reads_inside_workflows(pack, comfy_dirs):
    api = pack("api")
    workflows = comfy_dirs.user / "default" / "workflows"
    (workflows / "sub").mkdir()
    (workflows / "sub" / "wf.json").write_text("{}")

    assert api.LoadWorkflow().load("sub/wf.json") == ("{}",)


def test_load_workflow_filename_cannot_leave_workflows(pack, comfy_dirs):
    api = pack("api")
    (comfy_dirs.root / "secret.txt").write_text("secret")

    with pytest.raises(ValueError):
        api.LoadWorkflow().load("../../../secret.txt")


def test_symlink_inside_output_cannot_leave_output(lora, comfy_dirs):
    image = torch.zeros(1, 8, 8, 3)
    outside = comfy_dirs.root / "escaped_link"
    outside.mkdir()
    (comfy_dirs.output / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        lora.SaveImageWithText.execute(image, "1girl", "link")

    assert list(outside.iterdir()) == []
