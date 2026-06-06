"""Nodes in AlcheminePack/Model."""

import re
from pathlib import Path

import folder_paths
import comfy.sd
import comfy.utils


#################################################################
# Base class
#################################################################
class BaseModel:
    """Base class for Model nodes."""

    ...


#################################################################
# Nodes
#################################################################
class CachedLoraTagLoader(BaseModel):
    """Load LoRAs from `<lora:name:weight>` tags in the text, with result caching.

    Behaves like the original "Load LoRA Tag" node, but caches the patched
    (model, clip, prompt) outputs. As long as the input `text` and the input
    `model`/`clip` are unchanged, the cached result is returned directly,
    skipping LoRA re-loading and re-patching.
    """

    TAG_PATTERN = r"\<[0-9a-zA-Z\:\_\-\.\s\/\(\)\\\\]+\>"

    def __init__(self):
        # Cache of loaded lora files: {lora_path: lora_state_dict}
        self.loaded_loras = {}
        # Cache of the last patched result, keyed by (text, id(model), id(clip)).
        self.cache_key = None
        self.cache_value = None

    INPUT_TYPES = classmethod(
        lambda s: {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True}),
            }
        }
    )
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "STRING")
    FUNCTION = "load_lora"
    CATEGORY = "AlcheminePack/Model"

    def load_lora(self, model, clip, text):
        # Return the cached result when text and inputs are unchanged.
        key = (text, id(model), id(clip))
        if self.cache_key == key:
            return self.cache_value

        result = self._load_lora(model, clip, text)

        self.cache_key = key
        self.cache_value = result
        return result

    def _load_lora(self, model, clip, text):
        founds = re.findall(self.TAG_PATTERN, text)
        if len(founds) < 1:
            return (model, clip, text)

        model_lora = model
        clip_lora = clip

        lora_files = folder_paths.get_filename_list("loras")
        for f in founds:
            tag = f[1:-1]
            pak = tag.split(":")
            if pak[0] != "lora":
                continue
            if not (len(pak) > 1 and len(pak[1]) > 0):
                continue
            name = pak[1]

            w_model = w_clip = 0
            try:
                if len(pak) > 2 and len(pak[2]) > 0:
                    w_model = float(pak[2])
                    w_clip = w_model
                if len(pak) > 3 and len(pak[3]) > 0:
                    w_clip = float(pak[3])
            except ValueError:
                continue

            lora_name = None
            for lora_file in lora_files:
                if Path(lora_file).name.startswith(name) or lora_file.startswith(name):
                    lora_name = lora_file
                    break
            if lora_name is None:
                print(f"bypassed lora tag: {(name, w_model, w_clip)} >> {lora_name}")
                continue
            print(f"detected lora tag: {(name, w_model, w_clip)} >> {lora_name}")

            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = self.loaded_loras.get(lora_path)
            if lora is None:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_loras[lora_path] = lora

            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model_lora, clip_lora, lora, w_model, w_clip
            )

        plain_prompt = re.sub(self.TAG_PATTERN, "", text)
        return (model_lora, clip_lora, plain_prompt)
