# Base node(Save Image): https://github.com/comfyanonymous/ComfyUI/blob/a125cd84b054a57729b5eecab930ca9408719832/nodes.py#L1561

import os
import json
import hashlib
import threading
from pathlib import Path

import torch
import numpy as np
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageOps, ImageSequence

import node_helpers
import folder_paths
from comfy.cli_args import args


class AsyncSaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "io"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    @staticmethod
    def _save_image_in_thread(
        image,
        prompt,
        extra_pnginfo,
        full_output_folder,
        file,
        compress_level,
        disable_metadata,
    ):
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        metadata = None
        if not disable_metadata:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        img.save(
            os.path.join(full_output_folder, file),
            pnginfo=metadata,
            compress_level=compress_level,
        )

    def save_images(
        self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()

        for batch_number, image in enumerate(images):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"

            thread = threading.Thread(
                target=AsyncSaveImage._save_image_in_thread,
                args=(
                    image,
                    prompt,
                    extra_pnginfo,
                    full_output_folder,
                    file,
                    self.compress_level,
                    args.disable_metadata,
                ),
            )
            thread.start()

            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}


class PreviewLatestImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    },
                )
            },
        }

    CATEGORY = "io"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.prefix_append = ""

    def load_image(self, filename_prefix="ComfyUI"):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, None, None
            )
        )
        files = [
            f
            for f in Path(f"{full_output_folder}/{filename}").iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", "webp"]
        ]
        if files:
            image_path = max(files, key=lambda f: f.stat().st_ctime)
        else:
            raise FileNotFoundError(f"No valid image found in {full_output_folder}")

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            elif i.mode == "P" and "transparency" in i.info:
                mask = (
                    np.array(i.convert("RGBA").getchannel("A")).astype(np.float32)
                    / 255.0
                )
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, filename_prefix):
        m = hashlib.sha256()
        with open(filename_prefix, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    # @classmethod
    # def VALIDATE_INPUTS(s, image_path):
    #     if not folder_paths.exists_annotated_filepath(image):
    #         return "Invalid image file: {}".format(image)
    #     return True
