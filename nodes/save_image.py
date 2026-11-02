"""Nodes in AlcheminePack/Image."""

import io
import json
import os
import time
import zlib

import folder_paths
import torch
from PIL import Image
from comfy.cli_args import args

from .lib.utils import ensure_inside

# web/js/image_save.js reads these segments back.
METADATA_MARKER = b"\xff\xef"  # APP15
METADATA_SIGNATURE = b"ComfyUI\0"
METADATA_CHUNK_SIZE = 65533 - len(METADATA_SIGNATURE)
JPEG_QUALITY = 95
PREFIX_TIME_FORMATS = {
    "%year%": "%Y",
    "%month%": "%m",
    "%day%": "%d",
    "%hour%": "%H",
    "%minute%": "%M",
    "%second%": "%S",
}


def file_name(stem: str, counter: int) -> str:
    return f"{stem}_{counter:05}.jpg"


def next_counter(folder: str, stem: str) -> int:
    """Finds the end of the run of numbers from 1 without listing the folder."""

    def exists(counter):
        return os.path.exists(os.path.join(folder, file_name(stem, counter)))

    high = 1
    while exists(high):
        high *= 2
    low = high // 2
    while high - low > 1:
        middle = (low + high) // 2
        if exists(middle):
            low = middle
        else:
            high = middle
    return high


def metadata_segments(prompt, extra_pnginfo) -> bytes:
    """Compresses the metadata into JPEG segments, as many as its size needs."""
    metadata = dict(extra_pnginfo or {})
    if prompt is not None:
        metadata["prompt"] = prompt
    if args.disable_metadata or not metadata:
        return b""

    # One JSON string per key, like PNG text chunks: a prompt can hold NaN
    # (is_changed), which JSON.parse rejects, and must not take the workflow with it.
    strings = {key: json.dumps(value) for key, value in metadata.items()}
    payload = zlib.compress(json.dumps(strings).encode(), 1)
    segments = bytearray()
    for start in range(0, len(payload), METADATA_CHUNK_SIZE):
        chunk = payload[start : start + METADATA_CHUNK_SIZE]
        length = len(METADATA_SIGNATURE) + len(chunk) + 2
        segments += METADATA_MARKER + length.to_bytes(2, "big")
        segments += METADATA_SIGNATURE + chunk
    return bytes(segments)


class FastImageSave:
    """Save images as JPEG with the workflow embedded."""

    INPUT_TYPES = lambda: {
        "required": {
            "images": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "ComfyUI"}),
        },
        "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
    }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Image"
    OUTPUT_NODE = True

    @classmethod
    def execute(
        cls, images, filename_prefix: str = "ComfyUI", prompt=None, extra_pnginfo=None
    ) -> dict:
        output_dir = folder_paths.get_output_directory()
        filename_prefix = filename_prefix.replace(
            "%width%", str(images.shape[2])
        ).replace("%height%", str(images.shape[1]))
        now = time.localtime()
        for variable, time_format in PREFIX_TIME_FORMATS.items():
            filename_prefix = filename_prefix.replace(
                variable, time.strftime(time_format, now)
            )
        subfolder, filename = os.path.split(os.path.normpath(filename_prefix))
        folder = os.path.join(output_dir, subfolder)
        ensure_inside(output_dir, folder)
        os.makedirs(folder, exist_ok=True)
        counter = next_counter(folder, filename.replace("%batch_num%", "0"))

        segments = metadata_segments(prompt, extra_pnginfo)
        pixels = images.mul(255).clamp_(0, 255).to(torch.uint8).cpu().numpy()

        results = []
        for batch_number, image in enumerate(pixels):
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="JPEG", quality=JPEG_QUALITY)
            jpeg = buffer.getbuffer()
            # After SOI and the JFIF segment, which readers expect to come first.
            split = 4 + int.from_bytes(jpeg[4:6], "big")

            stem = filename.replace("%batch_num%", str(batch_number))
            # A gap can sit right before existing numbers: never overwrite them.
            while os.path.exists(os.path.join(folder, file_name(stem, counter))):
                counter += 1
            file = file_name(stem, counter)
            counter += 1
            with open(os.path.join(folder, file), "wb") as f:
                f.write(jpeg[:split])
                f.write(segments)
                f.write(jpeg[split:])
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
        return {"ui": {"images": results}, "result": (images,)}
