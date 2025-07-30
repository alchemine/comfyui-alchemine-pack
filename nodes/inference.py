import re
from io import BytesIO
from base64 import b64encode
from functools import lru_cache

import torch
import requests
import numpy as np
from PIL import Image

from .utils import get_logger, CONFIG

logger = get_logger()


#################################################################
# Base class
#################################################################
class BaseInference:
    """Base class for Inference nodes."""

    REQUEST_CACHE = {}

    @staticmethod
    def encode_image(image: torch.Tensor, return_bytes: bool = False) -> bytes | str:
        assert image.ndim == 4, "Image must be a 4D tensor"  # [B, H, W, C]
        image_tensor = image[0].detach().cpu().numpy()  # float in [0, 1]
        image_tensor = (image_tensor * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_tensor)
        image_bytes = BytesIO()
        image_pil.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()

        if return_bytes:
            return image_bytes
        else:
            image_b64 = b64encode(image_bytes).decode("utf-8")
            return image_b64


#################################################################
# Nodes
#################################################################
class GeminiInference(BaseInference):
    """Gemini inference."""

    INPUT_TYPES = lambda: {
        "required": {
            "system_instruction": (
                "STRING",
                {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                    "placeholder": "Prompt Text",
                    "dynamicPrompts": True,
                },
            ),
            "prompt": (
                "STRING",
                {
                    "default": "Hello, world!",
                    "multiline": True,
                    "placeholder": "Prompt Text",
                    "dynamicPrompts": True,
                },
            ),
            "gemini_api_key": ("STRING", {"default": ""}),
            "model": ("STRING", {"default": "latest"}),
            "max_output_tokens": ("INT", {"default": 100, "min": 1}),
            "seed": ("INT", {"default": 0, "min": 0}),
            "think": ("BOOLEAN", {"default": False}),
            "candidate_count": ("INT", {"default": 1, "min": 1}),
        },
        "optional": {
            "image": ("IMAGE", {"default": None}),
        },
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Inference"

    @classmethod
    def execute(
        cls,
        image: list[torch.Tensor] | None = None,
        system_instruction: str = "You are a helpful assistant.",
        prompt: str = "Hello, world!",
        gemini_api_key: str = "",
        model: str = "latest",
        max_output_tokens: int = 100,
        seed: int = 0,
        think: bool = False,
        candidate_count: int = 1,
    ) -> tuple[str]:
        # Add image tokens
        parts = []
        if image is not None:
            image_bytes = cls.encode_image(image, return_bytes=True)
            parts.append(
                {"inline_data": {"data": image_bytes, "mime_type": "image/png"}}
            )

        # Add response length limit
        prompt = f"{prompt}\n---\n결과는 {max_output_tokens//2}개의 단어 이내로 작성해주세요."
        parts.append({"text": prompt})

        # Caching request
        cache_key = (
            str(parts),
            system_instruction,
            gemini_api_key,
            model,
            max_output_tokens,
            seed,
            think,
            candidate_count,
        )
        if cache_key in cls.REQUEST_CACHE:
            return (cls.REQUEST_CACHE[cache_key],)

        # Generate response
        client = cls.get_client(gemini_api_key)
        response = client.models.generate_content(
            contents=[{"role": "user", "parts": parts}],
            **cls.get_config(
                client,
                system_instruction,
                model,
                max_output_tokens,
                seed,
                think=think,
                candidate_count=candidate_count,
            ),
        )

        # Check if all candidates are prohibited
        if response is None or response.candidates is None:
            valid_candidates = []
        else:
            PROHIBITED_CONTENT_REASON = "PROHIBITED_CONTENT"
            PROHIBITED_KEYWORDS = [
                "request",
                "cannot generate",
                "safety",
                "guideline",
                "refus",
                "sorry",
                "exploit",
                "abuse",
                "endanger child",
                "unholy",
                "unsafe",
                "policy",
                "violat",
                "harmless",
                "sexually suggestive",
                "explicit",
            ]
            valid_candidates = [
                c
                for c in response.candidates
                if c.finish_reason != PROHIBITED_CONTENT_REASON
                and all(
                    keyword not in c.content.parts[0].text.lower()
                    for keyword in PROHIBITED_KEYWORDS
                )
            ]

        warning_msg = f"{candidate_count-len(valid_candidates)}/{candidate_count} candidates are prohibited."
        if valid_candidates:
            # Select the first valid candidate
            valid_candidate = valid_candidates[0]
            result = valid_candidate.content.parts[0].text
            if result == "" and think and valid_candidate.finish_reason == "MAX_TOKENS":
                msg = f"More 'max_output_tokens' is needed to generate a response in thinking mode."
                logger.error(msg)
                raise ValueError(msg)
            logger.warning(warning_msg)
        else:
            # NOTE: Fallback to Gemini 2.0 Flash if all candidates are prohibited
            fallback_model = "models/gemini-2.0-flash"
            if model != fallback_model:
                # NOTE: Avoid the prohibited content by generating multiple candidates
                return cls.execute(
                    image=image,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    gemini_api_key=gemini_api_key,
                    model=fallback_model,
                    max_output_tokens=max_output_tokens,
                    seed=seed,
                    think=think,
                    candidate_count=8,  # 8 is the max
                )
            msg = f"No valid candidates found. {warning_msg}"
            logger.error(msg)
            raise ValueError(msg)

        cls.REQUEST_CACHE[cache_key] = result
        return (result,)

    @classmethod
    def IS_CHANGED(
        cls,
        image: list[torch.Tensor] | None = None,
        system_instruction: str = "You are a helpful assistant.",
        prompt: str = "Hello, world!",
        gemini_api_key: str = "",
        model: str = "latest",
        max_output_tokens: int = 100,
        seed: int = 0,
        think: bool = False,
        candidate_count: int = 1,
    ) -> tuple:
        return (
            image,
            system_instruction,
            prompt,
            gemini_api_key,
            model,
            max_output_tokens,
            seed,
            think,
            candidate_count,
        )

    @staticmethod
    @lru_cache
    def get_client(
        gemini_api_key: str = "",
    ):
        from google import genai

        if not gemini_api_key:
            try:
                gemini_api_key = CONFIG["inference"]["gemini_api_key"]
            except KeyError:
                raise ValueError(
                    "Gemini API key is not set. Please set the `gemini_api_key` in the `custom_nodes/comfyui-alchemine-pack/config.json` file."
                )
        return genai.Client(api_key=gemini_api_key)

    @classmethod
    @lru_cache
    def get_latest_model(cls, client) -> str:
        """Get the latest model."""
        GEMIMI_MODEL_NAME_PATTERN = r"^models/gemini-[\d\.]+-flash$"

        valid_models = []
        for model in client.models.list():
            if (
                re.match(GEMIMI_MODEL_NAME_PATTERN, model.name)
                and "generateContent" in model.supported_actions
                and model.name not in valid_models
            ):
                valid_models.append(model.name)
        return max(valid_models)

    @classmethod
    def get_config(
        cls,
        client,
        system_instruction: str,
        model: str = "latest",
        max_output_tokens: int = 100,
        seed: int = 0,
        think: bool = False,
        candidate_count: int = 1,
    ) -> dict:
        from google.genai.types import GenerateContentConfig, ThinkingConfig

        if model == "latest":
            model = cls.get_latest_model(client)

        return {
            "model": model,
            "config": GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=ThinkingConfig(thinking_budget=-1 if think else 0),
                max_output_tokens=max_output_tokens,
                seed=seed,
                candidate_count=candidate_count,
                # Fixed parameters
                stop_sequences=None,
                top_k=None,
                top_p=None,
                temperature=None,
                response_mime_type="text/plain",
            ),
        }


class OllamaInference(BaseInference):
    """Ollama inference."""

    INPUT_TYPES = lambda: {
        "required": {
            "system_instruction": (
                "STRING",
                {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                    "placeholder": "Prompt Text",
                    "dynamicPrompts": True,
                },
            ),
            "prompt": (
                "STRING",
                {
                    "default": "Hello, world!",
                    "multiline": True,
                    "placeholder": "Prompt Text",
                    "dynamicPrompts": True,
                },
            ),
            "ollama_url": ("STRING", {"default": ""}),
            "model": ("STRING", {"default": ""}),
            "max_output_tokens": ("INT", {"default": 100, "min": 1}),
            "seed": ("INT", {"default": 0, "min": 0}),
            "think": ("BOOLEAN", {"default": False}),
        },
        "optional": {
            "image": ("IMAGE", {"default": None}),
        },
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Inference"

    @classmethod
    def execute(
        cls,
        image: list[torch.Tensor] | None = None,
        system_instruction: str = "You are a helpful assistant.",
        prompt: str = "Hello, world!",
        ollama_url: str = "",
        model: str = "",
        max_output_tokens: int = 100,
        seed: int = 0,
        think: bool = False,
    ) -> tuple[str]:
        # Add image tokens
        user_message = {"role": "user"}

        # Add image content
        if image is not None:
            image_b64 = cls.encode_image(image)
            user_message["images"] = [image_b64]

        # Add text content
        user_message["content"] = prompt

        # Caching request
        cache_key = (
            str(user_message),
            system_instruction,
            ollama_url,
            model,
            max_output_tokens,
            seed,
            think,
        )
        if cache_key in cls.REQUEST_CACHE:
            return (cls.REQUEST_CACHE[cache_key],)

        # Check if the model is available
        ollama_url = ollama_url or CONFIG["inference"]["ollama_url"]
        response = requests.get(f"{ollama_url}/api/tags", timeout=0.5)
        response.raise_for_status()
        assert model in [
            m["name"] for m in response.json()["models"]
        ], f"Invalid model: {model}"

        # Generate response
        if model.startswith("qwen3"):
            # Optimal parameters for Qwen3
            options = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0}
        else:
            options = {}
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    user_message,
                ],
                "think": think,
                "stream": False,
                "options": {
                    "seed": seed,
                    "num_predict": max_output_tokens,
                    # "num_keep": 10,  # Keep model in memory for 10 minutes
                    **options,
                },
            },
        )
        response.raise_for_status()
        result = response.json()["message"]["content"]

        cls.REQUEST_CACHE[cache_key] = result
        return (result,)

    @classmethod
    def IS_CHANGED(
        cls,
        image: list[torch.Tensor] | None = None,
        system_instruction: str = "You are a helpful assistant.",
        prompt: str = "Hello, world!",
        ollama_url: str = "",
        model: str = "",
        max_output_tokens: int = 100,
        seed: int = 0,
        think: bool = False,
    ) -> tuple:
        return (
            image,
            system_instruction,
            prompt,
            ollama_url,
            model,
            max_output_tokens,
            seed,
            think,
        )


class TextEditingInference(BaseInference):
    """Text editing inference.

    References:
    - https://huggingface.co/grammarly/coedit-large (selected)
    - https://huggingface.co/vennify/t5-base-grammar-correction (backup)
    """

    INPUT_TYPES = lambda: {
        "required": {
            "predefined_system_instruction": (
                [
                    "",
                    "Fix the grammar",
                    "Make this text coherent",
                    "Rewrite to make this easier to understand",
                    "Paraphrase this",
                    "Write this more formally",
                    "Write in a more neutral way",
                ],
                {"default": "Fix the grammar"},
            ),
            "system_instruction": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "placeholder": "System Prompt Text",
                    "dynamicPrompts": True,
                },
            ),
            "prompt": (
                "STRING",
                {
                    "default": "When I grow up, I start to understand what he said is quite right.",
                    "multiline": True,
                    "placeholder": "Input Text",
                    "dynamicPrompts": True,
                },
            ),
            "seed": ("INT", {"default": 0, "min": 0}),
        }
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Inference"

    @classmethod
    def execute(
        cls,
        predefined_system_instruction: str = "Fix the grammar",
        system_instruction: str = "",
        prompt: str = "When I grow up, I start to understand what he said is quite right.",
        seed: int = 0,
    ) -> tuple[str]:
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        # Set seed
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

        # Caching request
        cache_key = (predefined_system_instruction, system_instruction, prompt, seed)
        if cache_key in cls.REQUEST_CACHE:
            edited_text = cls.REQUEST_CACHE[cache_key]
            torch.set_rng_state(rng_state)
            return (edited_text,)

        tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
        model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")
        input_text = f"{predefined_system_instruction or system_instruction}: {prompt}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length=256)
        edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Reset seed
        torch.set_rng_state(rng_state)

        cls.REQUEST_CACHE[cache_key] = edited_text
        return (edited_text,)

    @classmethod
    def IS_CHANGED(
        cls,
        predefined_system_instruction: str = "Fix the grammar",
        system_instruction: str = "",
        prompt: str = "When I grow up, I start to understand what he said is quite right.",
        seed: int = 0,
    ) -> tuple:
        return (predefined_system_instruction, system_instruction, prompt, seed)


if __name__ == "__main__":
    text = GeminiInference.execute(prompt="Hello, how are you?")
    print(text)
    # text = OllamaInference.execute(
    #     prompt="Hello, how are you?",
    #     model="qwen3:0.6b",
    #     think=True,
    #     max_output_tokens=1000,
    # )
    # print(text)
