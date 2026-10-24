import pytest
import torch
from PIL import Image


@pytest.fixture
def dataset(pack, comfy_dirs, request):
    """A folder of n images named 01.png, 02.png, ... under output."""

    def make(n):
        d = comfy_dirs.output / request.node.name
        d.mkdir()
        for i in range(1, n + 1):
            Image.new("RGB", (2, 2)).save(d / f"{i:02d}.png")
        return d.name

    return make


@pytest.fixture
def save(pack, comfy_dirs, request):
    """SaveImageWithText into a fresh folder; returns the saved relative paths."""
    node = pack("lora").SaveImageWithText
    dir_path = request.node.name

    def run(**kw):
        return node.execute(
            image=torch.zeros(1, 4, 4, 3),
            text=kw.pop("text", "a, b"),
            dir_path=dir_path,
            **kw,
        )

    run.dir = comfy_dirs.output / dir_path
    return run
