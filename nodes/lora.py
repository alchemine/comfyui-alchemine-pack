"""Nodes in AlcheminePack/Lora. (experimental nodes)"""

import os
from hashlib import md5
from pathlib import Path
from os.path import exists, relpath

import torch
import requests
import numpy as np
import folder_paths
from PIL import Image


#################################################################
# Base class
#################################################################
class BaseLora:
    """Base class for Lora nodes."""

    ...


#################################################################
# Nodes
#################################################################
class DownloadImage(BaseLora):
    """Download an image from url."""

    INPUT_TYPES = lambda: {
        "required": {
            "url": ("STRING",),
            "dir_path": ("STRING", {"default": "output/images"}),
        }
    }
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "file_path")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Lora"

    @classmethod
    def execute(cls, url: str, dir_path: str) -> tuple[str]:
        output_dir = Path(folder_paths.get_output_directory())
        dir_path = output_dir / dir_path
        if not exists(dir_path):
            os.makedirs(dir_path)

        extension = url.split(".")[-1]
        file_name = md5(url.encode()).hexdigest()[:8]
        idx = 1 + len(os.listdir(dir_path))
        file_path = dir_path / f"{idx}.{extension}"
        if exists(file_path):
            pil_image = Image.open(file_path).convert("RGB")
            image_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
            image = image_tensor.unsqueeze(0)
            return (image, file_path)

        response = requests.get(url)
        response.raise_for_status()
        image = response.content

        with open(file_path, "wb") as f:
            f.write(image)

        return (image, relpath(file_path, output_dir))

    @classmethod
    def IS_CHANGED(cls, url: str, dir_path: str) -> str:
        return url, dir_path


class SaveImageWithText(BaseLora):
    """Save an image with a text."""

    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "text": ("STRING",),
            "dir_path": ("STRING",),
            "prefix": ("STRING", {"default": ""}),
        }
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_path", "text_path")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Lora"

    @classmethod
    def execute(cls, image, text: str, dir_path: str, prefix: str = "") -> tuple[str]:
        output_dir = Path(folder_paths.get_output_directory())
        dir_path = output_dir / dir_path
        if not exists(dir_path):
            os.makedirs(dir_path)

        if prefix:
            idx = 1 + len(list(dir_path.glob(f"{prefix}_*.*"))) // 2
            prefix_path = dir_path / f"{prefix}_{idx}"
        else:
            idxs = {
                int(n.relative_to(dir_path).stem.split("_")[0])
                for n in dir_path.glob("[0-9]*")
            }
            idx = 1 + max(idxs) if idxs else 1
            prefix_path = dir_path / f"{idx}"
        text_path = prefix_path.with_suffix(".txt")
        image_path = prefix_path.with_suffix(".png")

        # Save text
        with open(text_path, "w") as f:
            f.write(text)

        # Save Image
        image_np = image.cpu().numpy()
        if len(image_np.shape) == 4:
            image_np = image_np[0]
        i = 255.0 * image_np
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img.save(image_path, compress_level=4)

        return (relpath(image_path, output_dir), relpath(text_path, output_dir))
