import os
import sys
import shutil
import tomllib

import pytest


@pytest.fixture
def train(pack, comfy_dirs, tmp_path):
    """TrainSdxlLora on `output/train/u59/v1/train/` with the given image count.

    sd-scripts is a shell script that writes the LoRA file and prints one
    progress line per step of `max_train_steps`, so the loss points the node
    sends can be counted.
    """
    lora = pack("lora")
    sys.modules["folder_paths"].base_path = str(tmp_path)
    sys.modules["folder_paths"].get_full_path = lambda *_: str(
        tmp_path / "ckpt.safetensors"
    )
    sd = tmp_path / "sd-scripts"
    (sd / ".venv" / "bin").mkdir(parents=True)
    (sd / "sdxl_train_network.py").touch()
    accelerate = sd / ".venv" / "bin" / "accelerate"
    accelerate.write_text(
        "#!/bin/sh\n"
        "cfg=$(printf '%s\\n' \"$@\" | sed -n 's/^--config_file=//p')\n"
        'out=$(sed -n \'s/^output_dir = "\\(.*\\)"/\\1/p\' "$cfg")\n'
        "total=$(sed -n 's/^max_train_steps = //p' \"$cfg\")\n"
        'mkdir -p "$out" && touch "$out/u59.safetensors"\n'
        'for i in $(seq 1 "$total"); do echo "steps: $i/$total [00:01<00:00, 1.0it/s, avr_loss=0.1]"; done\n'
    )
    accelerate.chmod(0o755)
    base = comfy_dirs.output / "train" / "u59" / "v1"
    shutil.rmtree(base, ignore_errors=True)
    (base / "train").mkdir(parents=True)
    sent = []
    sys.modules["server"].PromptServer.instance.send_sync = lambda event, data: (
        sent.append(data["step"]) if event == "alchemine.train_progress" else None
    )

    def run(images, **over):
        for i in range(images):
            (base / "train" / f"{i}.png").touch()
        _, run_dir = lora.TrainSdxlLora.execute(
            **{
                "unique_id": "59",
                "name": "u59",
                "version": "v1",
                "ckpt_name": "ckpt.safetensors",
                "train_unit": "steps",
                "train_length": 10,
                "batch_size": 1,
                "save_steps": 0,
                "sample_steps": 0,
                "logging_steps": 1,
                "learning_rate": 0.0001,
                "network_type": "LoRA",
                "r": 16,
                "lora_alpha": 8,
                "loraplus_lr_ratio": 0.0,
                "resolution": 1024,
                "optim": "AdamW8bit",
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.05,
                "shuffle_caption": True,
                "keep_tokens": 1,
                "gradient_checkpointing": False,
                "gradient_accumulation_steps": 1,
                "network_dropout": 0.0,
                "scale_weight_norms": 0.0,
                "noise_offset": True,
                "noise_offset_value": 0.0357,
                "seed": 42,
                "sample_prompts": "x --w 64 --h 64",
            }
            | over
        )
        config = tomllib.loads(
            open(os.path.join(run_dir, "training_config.toml")).read()
        )
        return config, sent

    return run
