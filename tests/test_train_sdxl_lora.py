import os
import re
import sys
import shutil
import tomllib

import pytest


@pytest.fixture
def configs(pack):
    lora = pack("lora")

    def _make(**over):
        kw = dict(
            images_path="/data/blony",
            ckpt_path="/models/ckpt.safetensors",
            output_name="blony",
            run_dir="/out/runs/blony_run",
            lora_dir="/loras/blony",
            mask_dir="",
            max_steps=1500,
            learning_rate=0.0001,
            network_type="LoRA",
            r=16,
            lora_alpha=8,
            loraplus_lr_ratio=0.0,
            resolution=1024,
            batch_size=1,
            optim="AdamW8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            shuffle_caption=True,
            keep_tokens=1,
            gradient_checkpointing=False,
            gradient_accumulation_steps=1,
            network_dropout=0.0,
            scale_weight_norms=0.0,
            noise_offset=True,
            noise_offset_value=0.0357,
            seed=42,
            save_steps=250,
            sample_steps=250,
            sample_prompts_path="/out/runs/blony_run/sample_prompts.txt",
        )
        kw.update(over)
        training, dataset = lora.sdxl_lora_configs(**kw)
        return tomllib.loads(training), tomllib.loads(dataset)

    return _make


def test_training_config_carries_the_model_network_and_schedule(configs):
    t, _ = configs()
    assert t["pretrained_model_name_or_path"] == "/models/ckpt.safetensors"
    assert t["output_dir"] == "/loras/blony" and t["output_name"] == "blony"
    assert (t["network_module"], t["network_dim"], t["network_alpha"]) == (
        "networks.lora",
        16,
        8,
    )
    assert (t["optimizer_type"], t["lr_scheduler"], t["learning_rate"]) == (
        "AdamW8bit",
        "cosine",
        0.0001,
    )
    assert t["max_train_steps"] == 1500 and t["lr_warmup_steps"] == 75


def test_fixed_settings_are_always_present(configs):
    t, _ = configs()
    assert (
        t["mixed_precision"] == "bf16"
        and t["sdpa"] is True
        and t["cache_latents"] is True
    )
    assert t["network_train_unet_only"] is True and t["log_with"] == "tensorboard"
    assert t["logging_dir"] == "/out/runs/blony_run/logs"


def test_snapshots_and_samples_are_on_by_default(configs):
    t, _ = configs()
    assert t["save_every_n_steps"] == 250 and t["sample_every_n_steps"] == 250
    assert (
        t["sample_prompts"] == "/out/runs/blony_run/sample_prompts.txt"
        and t["sample_sampler"] == "euler_a"
    )


def test_loss_weighting_is_fixed_and_regularizers_are_off_at_zero(configs):
    t, _ = configs()
    assert t["min_snr_gamma"] == 5 and t["noise_offset"] == 0.0357
    assert t["gradient_accumulation_steps"] == 1
    assert "network_dropout" not in t and "scale_weight_norms" not in t

    t, _ = configs(noise_offset=False)
    assert "noise_offset" not in t

    t, _ = configs(
        gradient_accumulation_steps=4, network_dropout=0.1, scale_weight_norms=1.0
    )
    assert t["gradient_accumulation_steps"] == 4
    assert t["network_dropout"] == 0.1 and t["scale_weight_norms"] == 1.0


def test_network_type_picks_the_sd_scripts_module(configs):
    assert configs()[0]["network_module"] == "networks.lora"
    assert configs(network_type="LoKr")[0]["network_module"] == "networks.lokr"


def test_prodigy_gets_its_safeguards_and_loraplus_goes_to_network_args(configs):
    t, _ = configs()
    assert "optimizer_args" not in t and "network_args" not in t

    t, _ = configs(optim="Prodigy", learning_rate=1.0, loraplus_lr_ratio=8.0)
    assert t["optimizer_type"] == "Prodigy" and t["learning_rate"] == 1.0
    assert t["optimizer_args"] == [
        "weight_decay=0.01",
        "betas=(0.9,0.99)",
        "safeguard_warmup=True",
        "use_bias_correction=True",
    ]
    assert t["network_args"] == ["loraplus_lr_ratio=8"]


def test_zero_turns_snapshots_and_samples_off(configs):
    t, _ = configs(save_steps=0, sample_steps=0)
    for key in (
        "save_every_n_steps",
        "sample_every_n_steps",
        "sample_prompts",
        "sample_sampler",
    ):
        assert key not in t


def test_dataset_config_reads_the_folder_in_place(configs):
    _, d = configs()
    subset = d["datasets"][0]["subsets"][0]
    assert subset == {"image_dir": "/data/blony", "num_repeats": 1}
    assert (
        d["datasets"][0]["resolution"] == 1024
        and d["general"]["caption_extension"] == ".txt"
    )


def test_caption_shuffle_and_keep_tokens_live_in_the_dataset_config(configs):
    _, d = configs(shuffle_caption=False, keep_tokens=2)
    assert d["general"]["shuffle_caption"] is False and d["general"]["keep_tokens"] == 2


def test_windows_paths_are_written_with_forward_slashes(configs):
    t, d = configs(ckpt_path="C:\\m\\ckpt.safetensors", images_path="C:\\data\\blony")
    assert t["pretrained_model_name_or_path"] == "C:/m/ckpt.safetensors"
    assert d["datasets"][0]["subsets"][0]["image_dir"] == "C:/data/blony"


@pytest.fixture
def fake_sd_scripts(pack, comfy_dirs, tmp_path):
    """A sd-scripts checkout next to ComfyUI whose accelerate only writes the LoRA file and one sample image where the config says."""
    lora = pack("lora")
    sys.modules["folder_paths"].base_path = str(tmp_path)
    sys.modules["folder_paths"].get_full_path = lambda *_: str(
        tmp_path / "ckpt.safetensors"
    )
    sys.modules["folder_paths"].get_filename_list = lambda *_: ["ckpt.safetensors"]
    sd = tmp_path / "sd-scripts"
    (sd / ".venv" / "bin").mkdir(parents=True)
    (sd / "sdxl_train_network.py").touch()
    accelerate = sd / ".venv" / "bin" / "accelerate"
    accelerate.write_text(
        "#!/bin/sh\n"
        "cfg=$(printf '%s\\n' \"$@\" | sed -n 's/^--config_file=//p')\n"
        'out=$(sed -n \'s/^output_dir = "\\(.*\\)"/\\1/p\' "$cfg")\n'
        'mkdir -p "$out/sample" && touch "$out/x.safetensors" "$out/sample/x_000010_00_20260101000000_42.png" "$out/sample/x_000010_01_20260101000001.png" "$out/sample/notes.txt"\n'
        'for i in 1 2 3 4 5 6 7 8 9 10; do echo "steps: $i/10 [00:01<00:00, 1.0it/s, avr_loss=0.1]"; done\n'
    )
    accelerate.chmod(0o755)

    def _run(name="x", version="v1", logging_steps=1, unique_id=None, **over):
        return lora.TrainSdxlLora.execute(
            unique_id=unique_id,
            name=name,
            version=version,
            ckpt_name="ckpt.safetensors",
            logging_steps=logging_steps,
            **{"train_unit": "epochs", "train_length": 10} | over,
            learning_rate=0.0001,
            network_type="LoRA",
            r=16,
            lora_alpha=8,
            loraplus_lr_ratio=0.0,
            resolution=1024,
            batch_size=1,
            optim="AdamW8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            shuffle_caption=True,
            keep_tokens=1,
            gradient_checkpointing=False,
            gradient_accumulation_steps=1,
            network_dropout=0.0,
            scale_weight_norms=0.0,
            noise_offset=True,
            noise_offset_value=0.0357,
            seed=42,
            save_steps=0,
            sample_steps=10,
            sample_prompts="x --w 64 --h 64",
        )

    return _run


@pytest.fixture
def dataset(comfy_dirs):
    """`output/train/x/v1/train/` with one image, fresh for each test."""
    base = comfy_dirs.output / "train" / "x" / "v1"
    shutil.rmtree(base, ignore_errors=True)
    (base / "train").mkdir(parents=True)
    (base / "train" / "1.png").touch()
    return base


def test_epochs_become_steps_from_the_image_count(fake_sd_scripts, dataset):
    (dataset / "train" / "2.jpg").touch()
    (dataset / "train" / "2.txt").touch()
    sent = []
    sys.modules["server"].PromptServer.instance.send_sync = lambda event, data: (
        sent.append(data) if event == "alchemine.train_start" else None
    )
    _, run_dir = fake_sd_scripts(unique_id="7")
    t = tomllib.loads(open(os.path.join(run_dir, "training_config.toml")).read())
    assert t["max_train_steps"] == 20 and t["lr_warmup_steps"] == 1
    assert sent[0]["images"] == 2 and sent[0]["steps"] == 20

    shutil.rmtree(dataset / "runs")
    _, run_dir = fake_sd_scripts(unique_id="7", train_unit="steps", train_length=7)
    t = tomllib.loads(open(os.path.join(run_dir, "training_config.toml")).read())
    assert t["max_train_steps"] == 7 and sent[-1]["steps"] == 7


def test_run_and_checkpoints_folders_share_one_timestamp(fake_sd_scripts, dataset):
    lora_path, run_dir = fake_sd_scripts()
    stamp = os.path.basename(run_dir)
    assert re.fullmatch(r"\d{8}_\d{6}", stamp)
    assert lora_path == str(dataset / "checkpoints" / stamp / "x.safetensors")


def test_the_node_refuses_a_version_without_a_train_folder(fake_sd_scripts, dataset):
    with pytest.raises(FileNotFoundError, match="x/v2/train"):
        fake_sd_scripts(version="v2")


def test_name_and_version_cannot_leave_the_train_folder(fake_sd_scripts, dataset):
    with pytest.raises(ValueError, match="outside"):
        fake_sd_scripts(name="../../x")


def test_the_lora_goes_to_checkpoints_and_the_rest_to_a_run_folder(
    fake_sd_scripts, dataset
):
    lora_path, run_dir = fake_sd_scripts()
    assert os.path.dirname(run_dir) == str(dataset / "runs")
    lora_dir = os.path.dirname(lora_path)
    assert os.listdir(lora_dir) == ["x.safetensors"]
    assert sorted(os.listdir(run_dir)) == [
        "dataset_config.toml",
        "sample",
        "sample_prompts.txt",
        "training_config.toml",
    ]
    assert sorted(os.listdir(os.path.join(run_dir, "sample"))) == [
        "notes.txt",
        "step10_20260101000000_00.png",
        "step10_20260101000001_01.png",
    ]
    t = tomllib.loads(open(os.path.join(run_dir, "training_config.toml")).read())
    d = tomllib.loads(open(os.path.join(run_dir, "dataset_config.toml")).read())
    assert t["output_dir"] == lora_dir and t["output_name"] == "x"
    assert t["logging_dir"] == run_dir + "/logs"
    assert d["datasets"][0]["subsets"][0]["image_dir"] == str(dataset / "train")


def test_a_mask_folder_next_to_train_turns_on_masked_loss(fake_sd_scripts, dataset):
    (dataset / "mask").mkdir()
    _, run_dir = fake_sd_scripts()
    t = tomllib.loads(open(os.path.join(run_dir, "training_config.toml")).read())
    d = tomllib.loads(open(os.path.join(run_dir, "dataset_config.toml")).read())
    assert t["masked_loss"] is True
    assert d["datasets"][0]["subsets"][0]["conditioning_data_dir"] == str(
        dataset / "mask"
    )


def test_without_a_mask_folder_the_configs_have_no_masked_loss(
    fake_sd_scripts, dataset
):
    _, run_dir = fake_sd_scripts()
    t = tomllib.loads(open(os.path.join(run_dir, "training_config.toml")).read())
    d = tomllib.loads(open(os.path.join(run_dir, "dataset_config.toml")).read())
    assert "masked_loss" not in t
    assert "conditioning_data_dir" not in d["datasets"][0]["subsets"][0]


def test_no_mask_dir_means_no_masked_loss(configs):
    t, d = configs()
    assert (
        "masked_loss" not in t
        and "conditioning_data_dir" not in d["datasets"][0]["subsets"][0]
    )


def test_mask_dir_turns_on_masked_loss_with_the_folder_as_conditioning_data(configs):
    t, d = configs(mask_dir="/data/blony_mask")
    assert t["masked_loss"] is True
    assert d["datasets"][0]["subsets"][0]["conditioning_data_dir"] == "/data/blony_mask"


def test_loss_every_n_steps_thins_the_progress_events_but_keeps_the_last(
    fake_sd_scripts, dataset
):
    sent = []
    sys.modules["server"].PromptServer.instance.send_sync = lambda event, data: (
        sent.append((event, data.get("step", data.get("model_dir"))))
    )
    _, run_dir = fake_sd_scripts(logging_steps=4, unique_id="7")
    stamp = os.path.basename(run_dir)
    assert sent == [
        ("alchemine.train_start", "output/train/x/v1/checkpoints/" + stamp),
        ("alchemine.train_progress", 4),
        ("alchemine.train_progress", 8),
        ("alchemine.train_progress", 10),
    ]
