import re
import json
import logging
import textwrap
from io import BytesIO
from time import sleep
from pathlib import Path
from base64 import b64encode
from functools import wraps, lru_cache

import torch
import requests
import numpy as np
from PIL import Image
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig


#################################################################
# Logger setup
#################################################################
ROOT_DIR = Path(__file__).parent.parent
CUSTOM_NODES_DIR = ROOT_DIR.parent
CONFIG = json.load(open(ROOT_DIR / "config.json"))


def get_logger(name: str = __file__, level: int = logging.WARNING):
    class RootNameFormatter(logging.Formatter):
        def format(self, record):
            record.name = str(Path(record.name).relative_to(CUSTOM_NODES_DIR))
            return super().format(record)

    logger = logging.getLogger(name)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = RootNameFormatter(
        "[%(asctime)s] [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger()


#################################################################
# Utility functions
#################################################################
def exception_handler(func):
    """Handle unexpected exceptions in a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.error(f"# Unexpected error in '{func.__name__}'", exc_info=True)
            raise

    return wrapper


def log_prompt(func):
    """Log prompt input and output in a Unicode box table with class name, showing all lines. Now uses thinner lines, adds Node row, and prevents prompt truncation with word wrapping."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        col_width1, col_width2 = [10, 100]

        def format_multiline(label: str, text: str) -> str:
            lines = text.splitlines() or [""]
            out = []
            first_row = True
            for line in lines:
                wrapped = textwrap.wrap(line, width=col_width2) or [""]
                for i, wline in enumerate(wrapped):
                    if first_row and i == 0:
                        row = f"│ {label:<{col_width1-2}} │ {wline.ljust(col_width2)} │"
                    else:
                        row = f"│ {'':<{col_width1-2}} │ {wline.ljust(col_width2)} │"
                    out.append(row)
                    first_row = False
            return "\n".join(out)

        # Prepare inputs
        node_label = args[0].__name__
        input_val = kwargs["text"]
        result = func(*args, **kwargs)
        output_val = result[0]

        # NOTE. 2: space for tags
        top = f"┌{'─'*col_width1}┬{'─'*(2+col_width2)}┐"
        mid = f"├{'─'*col_width1}┼{'─'*(2+col_width2)}┤"
        bot = f"└{'─'*col_width1}┴{'─'*(2+col_width2)}┘"

        # Prepare table content
        node_row = format_multiline("Node", node_label)
        before = format_multiline("Before", input_val)
        after = format_multiline("After", output_val)
        if len(result) > 1:
            filtered_tags = result[1]
            filtered = format_multiline("Filtered", filtered_tags)
            contents = [node_row, before, after, filtered]
        else:
            contents = [node_row, before, after]

        # Log
        content = f"\n{mid}\n".join(contents)
        table = f"{top}\n{content}\n{bot}"
        logger.debug(f"\n{table}")
        return result

    return wrapper


def retry(func, retries: int = 3, delay: float = 1e-2, exceptions=(Exception,)):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(retries):
            try:
                result = func(*args, **kwargs)
                break
            except exceptions:
                if attempt < retries - 1:
                    sleep(delay)
                else:
                    raise
        return result

    return wrapper


#################################################################
# Base class
#################################################################
class BaseInference:
    """Base class for Inference nodes."""

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
    """Gemini inference.

    TODO: fix 'think' argument
    TODO: should 'image_url' argument be needed?
    """

    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE", {"default": None}),
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
                    "default": "#1: They are standing",
                    "multiline": True,
                    "placeholder": "Prompt Text",
                    "dynamicPrompts": True,
                },
            ),
            "gemini_api_key": ("STRING", {"default": ""}),
            "model": ("STRING", {"default": "latest"}),
            "max_output_tokens": ("INT", {"default": 100, "min": 1}),
            "seed": ("INT", {"default": 0, "min": 0}),
            # "think": ("BOOLEAN", {"default": False}),
        }
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
        prompt: str = "",
        gemini_api_key: str = "",
        model: str = "latest",
        max_output_tokens: int = 100,
        seed: int = 0,
        # image_url: str | None = None,
        # think: bool = False,
    ) -> tuple[str]:
        # NOTE: Avoid the prohibited content by generating multiple candidates
        N_CANDIDATES = 8  # 8 is the max

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
                candidate_count=N_CANDIDATES,
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

        warning_msg = f"{N_CANDIDATES-len(valid_candidates)}/{N_CANDIDATES} candidates are prohibited."
        if valid_candidates:
            # Select the first valid candidate
            result = valid_candidates[0].content.parts[0].text
            logger.warning(warning_msg)
        else:
            # NOTE: Fallback to Gemini 2.0 Flash if all candidates are prohibited
            fallback_model = "models/gemini-2.0-flash"
            if model != fallback_model:
                return cls.execute(
                    image=image,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    gemini_api_key=gemini_api_key,
                    model=fallback_model,
                    max_output_tokens=max_output_tokens,
                    seed=seed,
                )

            msg = f"No valid candidates found. {warning_msg}"
            logger.error(msg)
            raise ValueError(msg)

        return (result,)

    @classmethod
    def IS_CHANGED(
        cls,
        image: list[torch.Tensor] | None = None,
        system_instruction: str = "You are a helpful assistant.",
        prompt: str = "",
        gemini_api_key: str = "",
        model: str = "latest",
        max_output_tokens: int = 100,
        seed: int = 0,
        # think: bool = False,
    ) -> tuple:
        return (
            image,
            system_instruction,
            prompt,
            gemini_api_key,
            model,
            max_output_tokens,
            seed,
            # think,
        )

    @staticmethod
    @lru_cache
    def get_client(
        gemini_api_key: str = "",
    ) -> genai.Client:
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
    def get_latest_model(cls, client: genai.Client) -> str:
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
        client: genai.Client,
        system_instruction: str,
        model: str = "latest",
        max_output_tokens: int = 100,
        seed: int = 0,
        candidate_count: int = 1,
        think: bool = False,
    ) -> dict:
        if model == "latest":
            model = cls.get_latest_model(client)

        return {
            "model": model,
            "config": GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=None if think else ThinkingConfig(thinking_budget=0),
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
            "image": ("IMAGE", {"default": None}),
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
                    "default": "#1: They are standing",
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
        }
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
        prompt: str = "",
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

        return (result,)

    @classmethod
    def IS_CHANGED(
        cls,
        image: list[torch.Tensor] | None = None,
        system_instruction: str = "You are a helpful assistant.",
        prompt: str = "",
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


if __name__ == "__main__":
    # text = GeminiInference.execute(prompt="Hello, how are you?")
    text = OllamaInference.execute(
        prompt="Hello, how are you?",
        model="qwen3:0.6b",
        think=True,
        max_output_tokens=1000,
    )
    print(text)
