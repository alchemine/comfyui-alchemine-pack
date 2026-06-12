"""Nodes in AlcheminePack/Model."""

import re

import folder_paths
from comfy.sd import load_lora_for_models
from comfy.utils import load_torch_file

from .lib.utils import get_logger

logger = get_logger(__file__)


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
    (model, clip). ComfyUI re-executes this node whenever the upstream prompt
    text changes, yet the LoRA patching only depends on the set of `<lora:...>`
    tags. So the patched (model, clip) is cached keyed by the lora signature
    plus the input model/clip identities: when only the surrounding prompt
    changes, the expensive LoRA re-patching is skipped while the plain-text
    STRING output is still refreshed for the current text.
    """

    TAG_PATTERN = r"<lora:.+?:[0-9.]+>"

    def __init__(self):
        # Cache of loaded lora files: {lora_path: lora_state_dict}
        self.loaded_loras = {}
        # Cache of the last patched (model, clip), keyed by
        # (lora_signature, id(model), id(clip)).
        self.cache_key = None
        self.cache_value = None

    @classmethod
    def extract_lora_tags(cls, text):
        """Extract the normalized set of `<lora:...>` tags from the text.

        Only `lora` tags (with a non-empty name) are kept; any surrounding
        prompt text and non-lora `<...>` tags are ignored. The result is a
        sorted tuple so two texts with the same lora tags compare equal even
        when the rest of the prompt differs or the tag order changes.
        """
        if not text:
            return ()

        tags = []
        for f in re.findall(cls.TAG_PATTERN, text):
            tag = f[1:-1]
            pak = tag.split(":")
            if pak[0] != "lora":
                continue
            if not (len(pak) > 1 and len(pak[1]) > 0):
                continue
            tags.append(tag)
        return tuple(sorted(tags))

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
        # Reuse the previously patched (model, clip) when the lora signature
        # and the input model/clip are unchanged, even if the surrounding
        # prompt text differs. The plain prompt is always recomputed so the
        # STRING output stays in sync with the current text.
        key = (self.extract_lora_tags(text), id(model), id(clip))
        if self.cache_key == key:
            model_lora, clip_lora = self.cache_value
        else:
            model_lora, clip_lora = self._patch_loras(model, clip, text)
            self.cache_key = key
            self.cache_value = (model_lora, clip_lora)

        plain_prompt = re.sub(self.TAG_PATTERN, "", text)
        return (model_lora, clip_lora, plain_prompt)

    def _patch_loras(self, model, clip, text):
        founds = re.findall(self.TAG_PATTERN, text)
        if len(founds) < 1:
            return (model, clip)

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

            w_model = float(pak[2])
            w_clip = w_model
            if len(pak) > 3 and len(pak[3]) > 0:
                w_clip = float(pak[3])

            lora_name = None
            for lora_file in lora_files:
                if lora_file.startswith(name):
                    lora_name = lora_file
                    break
            if lora_name is None:
                logger.debug(f"bypassed lora tag: {(name, w_model, w_clip)} >> {lora_name}")
                continue
            logger.info(f"detected lora tag: {(name, w_model, w_clip)} >> {lora_name}")

            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = self.loaded_loras.get(lora_path)
            if lora is None:
                lora = load_torch_file(lora_path, safe_load=True)
                self.loaded_loras[lora_path] = lora

            model_lora, clip_lora = load_lora_for_models(
                model_lora, clip_lora, lora, w_model, w_clip
            )

        return (model_lora, clip_lora)

    # NOTE: IS_CHANGED is not working for linked text
    # @classmethod
    # def IS_CHANGED(cls, model, clip, text):
    #     # Treat the node as unchanged as long as the set of lora tags and the
    #     # model/clip inputs are the same, regardless of the surrounding prompt
    #     # text. Changing model or clip forces re-execution.
    #     return model, clip, str(cls.extract_lora_tags(text))