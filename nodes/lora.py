"""Nodes in AlcheminePack/Lora. (experimental nodes)"""

import os
import math
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from os.path import exists, relpath

import torch
import requests
import numpy as np
import folder_paths
from PIL import Image, ImageOps
from server import PromptServer
from collections import Counter

from .lib.utils import ensure_inside, get_logger

_session = requests.Session()


#################################################################
# Base class
#################################################################
class BaseLora:
    """Base class for Lora nodes."""


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
    def execute(cls, url: str, dir_path: str) -> tuple:
        output_dir = Path(folder_paths.get_output_directory())
        dir_path = output_dir / dir_path
        ensure_inside(output_dir, dir_path)
        if not exists(dir_path):
            os.makedirs(dir_path)

        extension = url.split(".")[-1]
        idx = 1 + len(os.listdir(dir_path))
        file_path = dir_path / f"{idx}.{extension}"
        if not exists(file_path):
            response = _session.get(url)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)

        pil_image = Image.open(file_path).convert("RGB")
        image_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
        image = image_tensor.unsqueeze(0)
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
        },
        "optional": {
            "name": (
                "STRING",
                {
                    "forceInput": True,
                    "tooltip": "Source file name or path; its stem names the saved files instead of a running number.",
                },
            ),
            "overwrite": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "With name: rewrite files that already exist. Off, an existing pair is left as is.",
                },
            ),
        },
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_path", "text_path")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Lora"
    OUTPUT_NODE = True

    @classmethod
    def execute(
        cls,
        image,
        text: str,
        dir_path: str,
        prefix: str = "",
        name: str = "",
        overwrite: bool = False,
    ) -> tuple[str]:
        output_dir = Path(folder_paths.get_output_directory())
        dir_path = output_dir / dir_path
        ensure_inside(output_dir, dir_path)
        if not exists(dir_path):
            os.makedirs(dir_path)

        if name:
            prefix_path = dir_path / Path(name).stem
            ensure_inside(dir_path, prefix_path)
        elif prefix:
            ensure_inside(output_dir, dir_path / prefix)
            idx = 1 + len(list(dir_path.glob(f"{prefix}_*.*"))) // 2
            prefix_path = dir_path / f"{prefix}_{idx}"
        else:
            idxs = {
                int(n.relative_to(dir_path).stem.split("_")[0])
                for n in dir_path.glob("[0-9]*")
            }
            idx = 1 + max(idxs) if idxs else 1
            prefix_path = dir_path / f"{idx}"
        # Append instead of with_suffix: a stem like "a.000Z_1" has a dot of its own.
        text_path = dir_path / f"{prefix_path.name}.txt"
        image_path = dir_path / f"{prefix_path.name}.png"
        if name and not overwrite and text_path.exists() and image_path.exists():
            return (relpath(image_path, output_dir), relpath(text_path, output_dir))

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


class LoadImagesFromDir(BaseLora):
    """Load every image of a folder under the ComfyUI output directory, as a list."""

    INPUT_TYPES = lambda: {
        "required": {
            "dir_path": (
                "STRING",
                {
                    "default": "",
                    "tooltip": "Folder relative to the ComfyUI output directory.",
                },
            ),
        },
        "optional": {
            "count": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Images per run. 0 loads the whole folder.",
                },
            ),
            "chunk": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "control_after_generate": True,
                    "tooltip": "With count: which group of count images to load (0 = the first). Set it to increment and queue the workflow once per group.",
                },
            ),
        },
    }
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "names")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Lora"

    EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

    @classmethod
    def image_paths(cls, dir_path: str) -> list[Path]:
        output_dir = Path(folder_paths.get_output_directory())
        dir_path = output_dir / dir_path
        ensure_inside(output_dir, dir_path)
        if not dir_path.is_dir():
            return []
        return sorted(
            p for p in dir_path.iterdir() if p.suffix.lower() in cls.EXTENSIONS
        )

    @classmethod
    def chunk_paths(cls, dir_path: str, count: int, chunk: int) -> list[Path]:
        paths = cls.image_paths(dir_path)
        return paths[chunk * count : (chunk + 1) * count] if count else paths

    @classmethod
    def execute(
        cls, dir_path: str, count: int = 0, chunk: int = 0
    ) -> tuple[list[torch.Tensor], list[str]]:
        images, names = [], []
        for path in cls.chunk_paths(dir_path, count, chunk):
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            images.append(
                torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None]
            )
            names.append(path.name)
        return (images, names)

    @classmethod
    def IS_CHANGED(cls, dir_path: str, count: int = 0, chunk: int = 0) -> tuple:
        # The folder is usually filled by an upstream node (e.g. a downloader) in
        # the same run, so the node re-runs whenever its files change.
        return tuple(
            (p.name, p.stat().st_mtime) for p in cls.chunk_paths(dir_path, count, chunk)
        )


class FilterFrequentTags(BaseLora):
    """Drop the tags that appear in almost every caption of a dataset.

    A tag present in nearly all captions describes the subject itself, so it
    is better left out of the captions and learned into the trigger word.
    """

    INPUT_TYPES = lambda: {
        "required": {
            "captions": ("STRING", {"forceInput": True}),
            "threshold": (
                "FLOAT",
                {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
            ),
        },
        "optional": {
            "fixed_tags": ("STRING", {"default": ""}),
        },
    }
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("captions", "removed_tags")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Lora"

    @staticmethod
    def split(text: str) -> list[str]:
        return [t.strip() for t in text.split(",") if t.strip()]

    @classmethod
    def execute(
        cls, captions: list[str], threshold: list[float], fixed_tags: list[str] = None
    ) -> tuple:
        threshold = threshold[0]
        fixed = set(cls.split(fixed_tags[0])) if fixed_tags else set()
        tag_lists = [cls.split(c) for c in captions]
        counts = Counter(t for tags in tag_lists for t in set(tags))
        limit = threshold * len(tag_lists)
        removed = {t: n for t, n in counts.items() if n >= limit and t not in fixed}
        filtered = [
            ", ".join(t for t in tags if t not in removed) for tags in tag_lists
        ]
        report = ", ".join(
            f"{t}({n}/{len(tag_lists)})"
            for t, n in sorted(removed.items(), key=lambda x: (-x[1], x[0]))
        )
        return (filtered, report)


def sdxl_lora_configs(
    images_path: str,
    ckpt_path: str,
    output_name: str,
    run_dir: str,
    lora_dir: str,
    mask_dir: str,
    max_steps: int,
    learning_rate: float,
    network_type: str,
    r: int,
    lora_alpha: int,
    loraplus_lr_ratio: float,
    resolution: int,
    batch_size: int,
    optim: str,
    lr_scheduler_type: str,
    warmup_ratio: float,
    shuffle_caption: bool,
    keep_tokens: int,
    gradient_checkpointing: bool,
    gradient_accumulation_steps: int,
    network_dropout: float,
    scale_weight_norms: float,
    noise_offset: bool,
    noise_offset_value: float,
    seed: int,
    save_steps: int,
    sample_steps: int,
    sample_prompts_path: str,
) -> tuple[str, str]:
    """TOML texts (training config, dataset config) for sdxl_train_network.py."""
    q = lambda p: str(p).replace("\\", "/")
    training = f"""pretrained_model_name_or_path = "{q(ckpt_path)}"
output_dir = "{q(lora_dir)}"
output_name = "{output_name}"
save_model_as = "safetensors"
save_precision = "bf16"
logging_dir = "{q(Path(run_dir) / "logs")}"
log_with = "tensorboard"

network_module = "networks.{network_type.lower()}"
network_dim = {r}
network_alpha = {lora_alpha}
network_train_unet_only = true

optimizer_type = "{optim}"
learning_rate = {learning_rate:g}
lr_scheduler = "{lr_scheduler_type}"
lr_warmup_steps = {int(max_steps * warmup_ratio)}

max_train_steps = {max_steps}
train_batch_size = {batch_size}
gradient_accumulation_steps = {gradient_accumulation_steps}
min_snr_gamma = 5
mixed_precision = "bf16"
gradient_checkpointing = {str(gradient_checkpointing).lower()}
cache_latents = true
sdpa = true
max_data_loader_n_workers = 0
seed = {seed}
"""
    if mask_dir:
        training += "masked_loss = true\n"
    if optim == "Prodigy":
        # lr is a multiplier of the estimated step size, so warmup needs the safeguard
        training += 'optimizer_args = ["weight_decay=0.01", "betas=(0.9,0.99)", "safeguard_warmup=True", "use_bias_correction=True"]\n'
    if loraplus_lr_ratio > 0:
        training += f'network_args = ["loraplus_lr_ratio={loraplus_lr_ratio:g}"]\n'
    if network_dropout > 0:
        training += f"network_dropout = {network_dropout:g}\n"
    if scale_weight_norms > 0:
        training += f"scale_weight_norms = {scale_weight_norms:g}\n"
    if noise_offset:
        training += f"noise_offset = {noise_offset_value:g}\n"
    if save_steps > 0:
        training += f"save_every_n_steps = {save_steps}\n"
    if sample_steps > 0:
        training += f"""sample_every_n_steps = {sample_steps}
sample_prompts = "{q(sample_prompts_path)}"
sample_sampler = "euler_a"
"""
    dataset = f"""[general]
caption_extension = ".txt"
shuffle_caption = {str(shuffle_caption).lower()}
keep_tokens = {keep_tokens}
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
resolution = {resolution}
batch_size = {batch_size}

  [[datasets.subsets]]
  image_dir = "{q(images_path)}"
  num_repeats = 1
"""
    if mask_dir:
        dataset += f'  conditioning_data_dir = "{q(mask_dir)}"\n'
    return training, dataset


SAMPLE_PROMPT = (
    "1girl, solo --n worst quality, low quality --w 832 --h 1216 --s 28 --l 5 --d 42"
)
PROGRESS_RE = re.compile(r"(\d+)/(\d+) \[.*?avr_loss=([\d.]+)")
# sd-scripts sample file: <output_name>_<step>_<prompt index>_<timestamp>[_<seed>].png
SAMPLE_RE = re.compile(r".+_(\d{6})_(\d{2})_(\d{14})(?:_\d+)?\.png")


def parse_progress(line: str):
    """(step, total, loss) from an sd-scripts progress line, or None."""
    m = PROGRESS_RE.search(line)
    return (int(m[1]), int(m[2]), float(m[3])) if m else None


class TrainSdxlLora(BaseLora):
    """Train an SDXL LoRA with kohya sd-scripts installed next to ComfyUI.

    Everything lives under `output/train/<name>/<version>/`: `train/` holds the
    images with same-name `.txt` captions, read in place through a dataset
    config; `mask/`, when present, turns on masked loss; each training gets a
    timestamp folder under both `checkpoints/` (LoRA files) and `runs/`.
    sd-scripts runs from its own `.venv` as a child process and the node waits
    for it.
    """

    INPUT_TYPES = lambda: {
        "required": {
            "name": ("STRING", {"default": "my_lora"}),
            "version": ("STRING", {"default": "v1.0.0"}),
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            "train_unit": (["epochs", "steps"],),
            "train_length": ("INT", {"default": 20, "min": 1, "max": 100000}),
            "learning_rate": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.00001},
            ),
            "network_type": (["LoRA", "LoKr"],),
            "r": ("INT", {"default": 32, "min": 1, "max": 256}),
            "lora_alpha": ("INT", {"default": 16, "min": 1, "max": 256}),
            "loraplus_lr_ratio": (
                "FLOAT",
                {"default": 0.0, "min": 0.0, "max": 32.0, "step": 1.0},
            ),
            "resolution": (
                "INT",
                {"default": 1024, "min": 512, "max": 2048, "step": 64},
            ),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            "optim": (["Prodigy", "AdamWScheduleFree", "AdamW8bit", "AdamW"],),
            "lr_scheduler_type": (["cosine", "constant"],),
            "warmup_ratio": (
                "FLOAT",
                {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01},
            ),
            "shuffle_caption": ("BOOLEAN", {"default": True}),
            "keep_tokens": ("INT", {"default": 1, "min": 0, "max": 16}),
            "gradient_checkpointing": ("BOOLEAN", {"default": False}),
            "gradient_accumulation_steps": ("INT", {"default": 1, "min": 1, "max": 64}),
            "network_dropout": (
                "FLOAT",
                {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01},
            ),
            "scale_weight_norms": (
                "FLOAT",
                {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1},
            ),
            "noise_offset": ("BOOLEAN", {"default": True}),
            "noise_offset_value": (
                "FLOAT",
                {"default": 0.0357, "min": 0.0, "max": 0.2, "step": 0.0001},
            ),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            "save_steps": ("INT", {"default": 200, "min": 0, "max": 100000}),
            "sample_steps": ("INT", {"default": 200, "min": 0, "max": 100000}),
            "logging_steps": ("INT", {"default": 25, "min": 1, "max": 100000}),
            "sample_prompts": (
                "STRING",
                {"default": SAMPLE_PROMPT, "multiline": True},
            ),
        },
        "hidden": {"unique_id": "UNIQUE_ID"},
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_path", "run_dir")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Lora"

    @classmethod
    def execute(
        cls,
        name: str,
        version: str,
        ckpt_name: str,
        logging_steps: int,
        unique_id=None,
        **params,
    ) -> tuple[str, str]:
        name, version = name.strip(), version.strip()
        train_root = os.path.join(folder_paths.get_output_directory(), "train")
        base_dir = os.path.join(train_root, name, version)
        ensure_inside(train_root, base_dir)
        images_path = os.path.join(base_dir, "train")
        if not os.path.isdir(images_path):
            raise FileNotFoundError(f"images folder is missing: {images_path}")
        images = sum(
            f.lower().endswith(LoadImagesFromDir.EXTENSIONS)
            for f in os.listdir(images_path)
        )
        # One epoch is every image once; sd-scripts runs on a step count.
        batches = math.ceil(images / params["batch_size"])
        train_unit, train_length = params.pop("train_unit"), params.pop("train_length")
        max_steps = train_length * batches if train_unit == "epochs" else train_length
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if not ckpt_path:
            raise FileNotFoundError(f"checkpoint not found: {ckpt_name}")
        sd_scripts_path = os.path.join(folder_paths.base_path, "sd-scripts")
        script = os.path.join(sd_scripts_path, "sdxl_train_network.py")
        accelerate = os.path.join(sd_scripts_path, ".venv", "bin", "accelerate")
        for path in (script, accelerate):
            if not os.path.exists(path):
                raise FileNotFoundError(f"sd-scripts is not installed: {path}")

        # One timestamp names both folders of a training, so its configs, logs,
        # samples and LoRA files are found side by side.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, "runs", stamp)
        lora_dir = os.path.join(base_dir, "checkpoints", stamp)
        mask_dir = os.path.join(base_dir, "mask")
        if not os.path.isdir(mask_dir):
            mask_dir = ""
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(lora_dir, exist_ok=True)
        if unique_id is not None:
            PromptServer.instance.send_sync(
                "alchemine.train_start",
                {
                    "node": unique_id,
                    "model_dir": os.path.join(
                        "output",
                        os.path.relpath(lora_dir, folder_paths.get_output_directory()),
                    ),
                    "images": images,
                    "steps": max_steps,
                },
            )

        sample_prompts_path = os.path.join(run_dir, "sample_prompts.txt")
        with open(sample_prompts_path, "w", encoding="utf-8") as f:
            f.write(params.pop("sample_prompts"))
        training, dataset = sdxl_lora_configs(
            images_path,
            ckpt_path,
            name,
            run_dir,
            lora_dir,
            mask_dir,
            sample_prompts_path=sample_prompts_path,
            max_steps=max_steps,
            **params,
        )
        config_path = os.path.join(run_dir, "training_config.toml")
        dataset_path = os.path.join(run_dir, "dataset_config.toml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(training)
        with open(dataset_path, "w", encoding="utf-8") as f:
            f.write(dataset)

        cmd = [
            accelerate,
            "launch",
            "--num_cpu_threads_per_process=2",
            script,
            f"--config_file={config_path}",
            f"--dataset_config={dataset_path}",
        ]
        env = os.environ | {"PYTHONIOENCODING": "utf-8"}
        log = get_logger()
        log.info("[TrainSdxlLora] %s", " ".join(cmd))
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=sd_scripts_path,
            env=env,
        ) as proc:
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                print(f"[sd-scripts] {line}", flush=True)  # noqa: T201
                progress = parse_progress(line)
                if progress and unique_id is not None:
                    step, total, loss = progress
                    if step % logging_steps and step != total:
                        continue
                    PromptServer.instance.send_sync(
                        "alchemine.train_progress",
                        {"node": unique_id, "step": step, "total": total, "loss": loss},
                    )
        if proc.returncode != 0:
            raise RuntimeError(
                f"sd-scripts exited with code {proc.returncode}, see {run_dir}"
            )

        lora_path = os.path.join(lora_dir, f"{name}.safetensors")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"training finished but {lora_path} is missing")
        # sd-scripts writes samples next to the LoRA files; keep them with the
        # run as step<step>_<timestamp>_<prompt line>.png
        sample_dir = os.path.join(lora_dir, "sample")
        if os.path.isdir(sample_dir):
            for file in os.listdir(sample_dir):
                m = SAMPLE_RE.fullmatch(file)
                if m:
                    renamed = f"step{int(m[1])}_{m[3]}_{m[2]}.png"
                    os.rename(
                        os.path.join(sample_dir, file),
                        os.path.join(sample_dir, renamed),
                    )
            shutil.move(sample_dir, os.path.join(run_dir, "sample"))
        return (lora_path, run_dir)
