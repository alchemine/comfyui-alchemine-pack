"""Nodes in AlcheminePack/Inference.

A single `OpenAIInference` node covers every OpenAI-compatible backend
(OpenAI, vLLM, Ollama's `/v1` endpoint, Gemini's OpenAI-compatible endpoint,
etc.) — just point `base_url`/`api_key`/`model` at the desired server.
"""

import os
import re
from io import BytesIO
from base64 import b64encode

import torch
import requests
import numpy as np
from PIL import Image

from .lib.utils import get_logger


logger = get_logger()


CACHE_MAX_SIZE = 10

# Upper bound for the `max_output_tokens` widget. ComfyUI defaults an INT
# widget without an explicit `max` to 2048, which silently caps long outputs;
# raise it so large context models can return full-length responses.
MAX_OUTPUT_TOKENS_LIMIT = 131072

# `requests` blocks forever without a timeout, so a wrong/unreachable `base_url`
# would hang the node indefinitely. Use a short *connect* timeout to fail fast
# on a bad host, but leave the *read* timeout open (None) so legitimately long
# generations are never cut off. No retries — surface the error immediately.
CONNECT_TIMEOUT = 1


#################################################################
# Base class
#################################################################
class BaseInference:
    """Base class for Inference nodes."""

    # NOTE: Cache for API requests
    REQUEST_CACHE = {}

    @classmethod
    def _set_cache(cls, key, value) -> None:
        """Set cache with LRU eviction."""
        cls.REQUEST_CACHE[key] = value
        if len(cls.REQUEST_CACHE) > CACHE_MAX_SIZE:
            first_item_key = next(iter(cls.REQUEST_CACHE))
            del cls.REQUEST_CACHE[first_item_key]

    @staticmethod
    def encode_image(image: torch.Tensor, return_bytes: bool = False) -> bytes | str:
        """Encode image to bytes or base64.

        Args:
            image (torch.Tensor): Image tensor.
                Must be a 4D tensor in the shape of [B, H, W, C].
            return_bytes (bool): Whether to return bytes or base64.
                If True, return bytes.
                If False, return base64.

        Returns:
            bytes | str: Encoded image.
        """
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
class OpenAIInference(BaseInference):
    """Inference with any OpenAI-compatible API.

    Works with OpenAI, vLLM, Ollama (`/v1`), and Gemini's OpenAI-compatible
    endpoint — set `base_url`/`api_key`/`model` to the backend you want.

    Args:
        system_instruction (str): System instruction.
        prompt (str): Prompt.
        base_url (str): API base URL (e.g. https://api.openai.com/v1). Falls back to `OPENAI_BASE_URL` in `.env`.
        api_key (str): API key. Falls back to `OPENAI_API_KEY` in `.env`.
        model (str | None): Model name (e.g. gpt-4o, gpt-4o-mini). If None, auto-detected from /models endpoint (uses the model if only one is available).
        max_output_tokens (int): Maximum output tokens.
        seed (int): Seed.
        temperature (float): Sampling temperature.
        think (bool): Whether to use thinking mode.
        image (list[torch.Tensor] | None): Image tensor.
            Must be a 4D tensor in the shape of [B, H, W, C].

    Returns:
        tuple[str]: Response.
    """

    INPUT_TYPES = lambda: {
        "required": {
            "prompt": (
                "STRING",
                {
                    "default": "Hello, world!",
                    "multiline": True,
                    "placeholder": "Prompt Text",
                    "dynamicPrompts": True,
                },
            ),
            "system_instruction": (
                "STRING",
                {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                    "placeholder": "Prompt Text",
                    "dynamicPrompts": True,
                },
            ),
            "base_url": ("STRING", {"default": ""}),
            "api_key": ("STRING", {"default": ""}),
            "model": ("STRING", {"default": ""}),
            "max_output_tokens": (
                "INT",
                {"default": 100, "min": 1, "max": MAX_OUTPUT_TOKENS_LIMIT},
            ),
            "seed": ("INT", {"default": 0, "min": 0}),
            "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
            "think": ("BOOLEAN", {"default": False}),
        },
        "optional": {
            "image": ("IMAGE", {"default": None}),
        },
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "reasoning")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Inference"

    @classmethod
    def execute(
        cls,
        prompt: str = "Hello, world!",
        system_instruction: str = "You are a helpful assistant.",
        image: list[torch.Tensor] | None = None,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        max_output_tokens: int = 100,
        seed: int = 0,
        temperature: float = 0.7,
        think: bool = False,
    ) -> tuple[str, str]:
        # Resolve config: node inputs take priority, then fall back to `.env`.
        base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not base_url:
            raise ValueError(
                "base_url is not set. Provide it on the node or set "
                "OPENAI_BASE_URL in `.env`."
            )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # Auto-detect model from /models endpoint if not specified
        if not model:
            model = cls._get_model(base_url, headers)

        # Build user message content
        if image is not None:
            image_b64 = cls.encode_image(image)
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {"type": "text", "text": prompt},
            ]
        else:
            user_content = prompt

        # Caching request
        cache_key = (
            str(user_content),
            system_instruction,
            base_url,
            api_key,
            model,
            max_output_tokens,
            seed,
            temperature,
            think,
        )
        if cache_key in cls.REQUEST_CACHE:
            return cls.REQUEST_CACHE[cache_key]

        # Generate response
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_output_tokens,
            "seed": seed,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": think},
        }
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=(CONNECT_TIMEOUT, None),
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]

        # Reasoning is exposed either as a dedicated `reasoning_content` field
        # (vLLM, DeepSeek, etc.) or inlined as a <think>...</think> block in the
        # content. Pull it out so the answer and the reasoning are separate.
        result = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        if not reasoning:
            result, reasoning = cls._split_think(result)

        cls._set_cache(cache_key, (result, reasoning))
        return (result, reasoning)

    @staticmethod
    def _split_think(content: str) -> tuple[str, str]:
        """Split an inlined `<think>...</think>` block out of the content.

        Returns (answer, reasoning); reasoning is "" when no block is present.
        """
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if not match:
            return content, ""
        reasoning = match.group(1).strip()
        answer = (content[: match.start()] + content[match.end() :]).strip()
        return answer, reasoning

    @staticmethod
    def _get_model(base_url: str, headers: dict) -> str:
        """Auto-detect model from /models endpoint.

        If only one model is available, use it.
        Otherwise, raise an error with available models.
        """
        response = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers=headers,
            timeout=(CONNECT_TIMEOUT, 30),
        )
        response.raise_for_status()
        models = [m["id"] for m in response.json()["data"]]
        if len(models) == 1:
            logger.info(f"Auto-detected model: {models[0]}")
            return models[0]
        msg = f"Multiple models available: {models}. Please specify one in the 'model' field."
        logger.error(msg)
        raise ValueError(msg)

    @classmethod
    def IS_CHANGED(
        cls,
        prompt: str = "Hello, world!",
        system_instruction: str = "You are a helpful assistant.",
        image: list[torch.Tensor] | None = None,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        max_output_tokens: int = 100,
        seed: int = 0,
        temperature: float = 0.7,
        think: bool = False,
    ) -> tuple:
        return (
            prompt,
            system_instruction,
            image,
            base_url,
            api_key,
            model,
            max_output_tokens,
            seed,
            temperature,
            think,
        )


if __name__ == "__main__":
    text = OpenAIInference.execute(prompt="Hello, how are you?")
    print(text)
