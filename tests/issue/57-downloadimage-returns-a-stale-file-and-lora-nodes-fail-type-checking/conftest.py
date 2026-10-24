import io
import types
import zlib

import pytest
from PIL import Image


def color_of(url):
    """A distinct color per URL, so a returned image shows which URL it came from."""
    c = zlib.crc32(url.encode())
    return (c & 255, c >> 8 & 255, c >> 16 & 255)


@pytest.fixture
def download(pack, comfy_dirs, monkeypatch, request):
    """DownloadImage into a fresh folder; records every URL actually fetched."""
    module = pack("lora")
    fetched = []

    def get(url):
        fetched.append(url)
        buf = io.BytesIO()
        Image.new("RGB", (2, 2), color_of(url)).save(buf, format="PNG")
        return types.SimpleNamespace(
            content=buf.getvalue(), raise_for_status=lambda: None
        )

    monkeypatch.setattr(module._session, "get", get)
    dir_path = request.node.name

    def run(url):
        image, file_path = module.DownloadImage.execute(url, dir_path)
        return image, comfy_dirs.output / file_path

    run.dir = comfy_dirs.output / dir_path
    run.color_of = color_of
    run.pixel = pixel
    run.fetched = fetched
    return run


def pixel(image):
    return tuple(round(v * 255) for v in image[0, 0, 0].tolist())
