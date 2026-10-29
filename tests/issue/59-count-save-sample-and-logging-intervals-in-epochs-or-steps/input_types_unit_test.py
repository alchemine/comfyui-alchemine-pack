import sys


def test_the_unit_inputs_are_optional_and_default_to_steps(pack):
    lora = pack("lora")
    sys.modules["folder_paths"].get_filename_list = lambda *_: ["ckpt.safetensors"]
    inputs = lora.TrainSdxlLora.INPUT_TYPES()
    for key in ("save", "sample", "logging"):
        assert f"{key}_unit" not in inputs["required"]
        assert inputs["optional"][f"{key}_unit"] == (
            ["epochs", "steps"],
            {"default": "steps"},
        )
