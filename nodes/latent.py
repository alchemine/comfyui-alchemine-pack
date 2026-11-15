"""Nodes in AlcheminePack/Latent."""

import torch
import comfy.model_management


DIMENSIONS = [
    "512 x 512 (1:1)",
    "768 x 512 (1.5:1)",
    "960 x 512 (1.875:1)",
    "1024 x 512 (2:1)",
    "1024 x 576 (1.778:1)",
    "1536 x 640 (2.4:1)",
    "1344 x 768 (1.75:1)",
    "1216 x 832 (1.46:1)",
    "1152 x 896 (1.286:1)",
    "1024 x 1024 (1:1)",
]


class LatentWithSeed:
    """Empty latents split into a list of single images, each with its own seed.

    Item i gets seed + i and keeps batch_index [i], so sampling it gives the
    same image as item i of an unsplit batch would with that seed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "dimensions": (DIMENSIONS, {"default": "1024 x 1024 (1:1)"}),
                "invert": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "scale_by": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("LATENT", "seed")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Latent"

    @classmethod
    def execute(cls, seed, dimensions, invert, batch_size, scale_by):
        width, height = (int(x) for x in dimensions.split(" (")[0].split(" x "))
        if invert:
            width, height = height, width
        latent_width = round(width // 8 * scale_by)
        latent_height = round(height // 8 * scale_by)
        latents = [
            {
                "samples": torch.zeros(
                    [1, 4, latent_height, latent_width],
                    device=comfy.model_management.intermediate_device(),
                    dtype=comfy.model_management.intermediate_dtype(),
                ),
                "batch_index": [i],
            }
            for i in range(batch_size)
        ]
        return (latents, [seed + i for i in range(batch_size)])
