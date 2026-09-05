"""Nodes in AlcheminePack/API."""

import asyncio
import hashlib
import io
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import folder_paths
import numpy as np
import requests
import torch
from PIL import Image

from comfy_execution.graph import ExecutionBlocker

from .lib.utils import any_typ, get_logger
from .lib import joblock

logger = get_logger()

# This module's slot in the shared `jobs.lock` (the Grok nodes use "grok").
_KIND = "api"

# A job lives in the shared lock keyed by `_KIND`. Present (with a prompt_id) =>
# a job is in progress / awaiting collection; absent => free to submit. The
# read-modify-write is guarded by `joblock.guard` so concurrent submits can't
# both pass the "is it free?" check at the same instant.
_lock_guard = joblock.guard

# If an in-flight job is older than this and the remote has no record of it
# (e.g. the server was replaced), treat the slot as free so submits aren't stuck
# forever.
_STALE_SEC = 600


def _read_lock() -> "dict | None":
    return joblock.read_lock(_KIND)


def _write_lock(record: "dict | None") -> None:
    joblock.write_lock(_KIND, record)


def _workflows_dir() -> str:
    user_dir = getattr(folder_paths, "get_user_directory", lambda: folder_paths.user_directory)()
    return os.path.join(user_dir, "default", "workflows")


#################################################################
# Client
#################################################################
_FETCH_CONCURRENCY = 16


class ApiComfyClient:
    def __init__(self, base_url: str, api_key: str | None = None, pool_size: int = 32):
        self.base = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self.session = requests.Session()
        retry = requests.adapters.Retry(
            total=5,
            connect=5,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
        )
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retry
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def headers(self):
        return self.session.headers

    def submit(self, api_workflow: dict) -> str:
        r = self.session.post(
            f"{self.base}/prompt",
            json={"prompt": api_workflow, "client_id": self.client_id},
            headers=self.headers,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["prompt_id"]

    def wait(self, prompt_id: str, timeout: int = 600, interval: float = 1.5) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = self.session.get(
                f"{self.base}/history/{prompt_id}",
                headers=self.headers,
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            if prompt_id in data:
                entry = data[prompt_id]
                status = entry.get("status", {})
                if status.get("completed"):
                    return entry["outputs"]
                if status.get("status_str") == "error":
                    raise RuntimeError(f"Remote workflow failed: {status}")
            time.sleep(interval)
        raise TimeoutError(f"Remote generation timed out after {timeout}s")

    def fetch_image(self, filename: str, subfolder: str, type_: str) -> bytes:
        last_exc = None
        for attempt in range(4):
            try:
                r = self.session.get(
                    f"{self.base}/view",
                    params={"filename": filename, "subfolder": subfolder, "type": type_},
                    timeout=60,
                )
                r.raise_for_status()
                return r.content
            except requests.exceptions.ConnectionError as e:
                last_exc = e
                time.sleep(0.5 * (attempt + 1))
        raise last_exc

    def upload_image(self, png_bytes: bytes, filename: str) -> str:
        """POST to `/upload/image` and return the server-assigned filename."""
        files = {"image": (filename, png_bytes, "image/png")}
        data = {"overwrite": "true", "type": "input"}
        r = self.session.post(
            f"{self.base}/upload/image",
            files=files,
            data=data,
            headers=self.headers,
            timeout=120,
        )
        r.raise_for_status()
        body = r.json()
        return body.get("name", filename)


#################################################################
# Helpers
#################################################################
def _load_workflow(s: str) -> dict:
    s = s.strip()
    if not s:
        raise ValueError("workflow_json is empty")
    if os.path.isfile(s):
        with open(s, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(s)


_TEXT_KEY_CANDIDATES = ("text", "prompt", "string", "positive", "caption")


def _inject_text(wf: dict, node_id: str, text: str) -> None:
    if node_id not in wf:
        raise KeyError(f"Node id {node_id!r} not in workflow")
    inputs = wf[node_id]["inputs"]
    for key in _TEXT_KEY_CANDIDATES:
        if isinstance(inputs.get(key), str):
            inputs[key] = text
            return
    raise KeyError(
        f"Node {node_id!r} has no string input among {_TEXT_KEY_CANDIDATES}; "
        f"existing inputs: {list(inputs.keys())}"
    )


def _inject_seed(wf: dict, node_id: str, seed: int) -> None:
    if node_id not in wf:
        raise KeyError(f"Seed node id {node_id!r} not in workflow")
    inputs = wf[node_id]["inputs"]
    for key in ("seed", "noise_seed"):
        if key in inputs:
            inputs[key] = int(seed)
            return
    raise KeyError(f"Node {node_id!r} has no 'seed' or 'noise_seed' input")


def _tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    """Convert a ComfyUI IMAGE tensor `(N,H,W,3)` (first frame) to PNG bytes."""
    if image.ndim == 4:
        image = image[0]
    arr = (image.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _inject_image(wf: dict, node_id: str, filename: str) -> None:
    if node_id not in wf:
        raise KeyError(f"Image node id {node_id!r} not in workflow")
    wf[node_id]["inputs"]["image"] = filename


def _restore_stripped_braces(s: str) -> str:
    """Heuristically restore `{`/`}` stripped by wildcard-style text processors.
    Indent-stack based: lines ending in `:` open an object, dedents close objects.
    Comma-only lines mark a close + trailing comma."""
    lines = s.split("\n")
    result: list[str] = []
    indent_stack = [0]
    pending_open = False  # last appended line ends with `:` and expects `{`

    def close_to(target_indent: int, with_comma: bool = False) -> None:
        while len(indent_stack) > 1 and indent_stack[-1] > target_indent:
            popped = indent_stack.pop()
            result.append(" " * popped + "}")
        if with_comma and result and result[-1].endswith("}"):
            result[-1] += ","

    for raw in lines:
        stripped = raw.lstrip()
        if not stripped:
            continue
        indent = len(raw) - len(stripped)
        content = stripped.rstrip()

        if content == ",":
            close_to(indent, with_comma=True)
            continue

        if pending_open:
            result[-1] += " {"
            indent_stack.append(indent)
            pending_open = False
        elif indent < indent_stack[-1]:
            close_to(indent)

        result.append(" " * indent + content)
        if content.endswith(":"):
            pending_open = True

    while len(indent_stack) > 1:
        popped = indent_stack.pop()
        result.append(" " * popped + "}")

    return "{\n" + "\n".join(result) + "\n}"


def _apply_overrides(wf: dict, overrides_str: str) -> None:
    """Replace entire node entries in `wf` with the dicts provided in
    `overrides_str` (JSON object: `{node_id: <node_dict>}`)."""
    s = overrides_str.strip()
    if not s:
        return
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        # Heuristic recovery: upstream wildcard processor may have stripped
        # `{`/`}`. Try restoring and reparse.
        try:
            restored = _restore_stripped_braces(s)
            data = json.loads(restored)
            logger.info("[ApiGenerate] overrides: recovered from wildcard-stripped braces")
        except json.JSONDecodeError as e2:
            preview = s[:200].replace("\n", "\\n")
            raise ValueError(
                f"overrides must be valid JSON: {e2}\n"
                f"--- received (first 200 chars) ---\n{preview}"
            )
    if not isinstance(data, dict):
        raise ValueError("overrides must be a JSON object (dict)")
    for node_id, node in data.items():
        if not isinstance(node, dict):
            raise ValueError(f"overrides[{node_id!r}] must be a dict")
        wf[str(node_id)] = node
    logger.info(f"[ApiGenerate] overrode {len(data)} node(s): {list(data.keys())}")


def _validate_api_url(api_url: str) -> None:
    if not re.fullmatch(r"https?://[^\s]+", api_url.strip()):
        raise ValueError(
            "api_url은 http(s)://host[:port][/path] 와 같은 형식이어야 합니다. "
            "예: https://xxxx-8188.proxy.runpod.net/ 또는 http://127.0.0.1:8188"
        )


def _prepare_and_submit(
    client: ApiComfyClient,
    workflow: str,
    positive_prompt: str,
    positive_prompt_id: str,
    negative_prompt_id: str,
    seed: int,
    seed_id: str,
    image_node_id: str,
    negative_prompt: str,
    image: "torch.Tensor | None",
    overrides: str,
) -> str:
    """Build the API workflow (inject prompt/seed/image/overrides), upload any
    image, submit to the remote, and return the `prompt_id`. Shared by the
    blocking `ApiGenerate` and the fire-and-forget `ApiSubmit`."""
    wf = _load_workflow(workflow)
    _inject_text(wf, positive_prompt_id, positive_prompt)
    if negative_prompt:
        if not negative_prompt_id:
            raise ValueError("`negative_prompt_id` is required when `negative_prompt` is provided")
        _inject_text(wf, negative_prompt_id, negative_prompt)
        logger.info(f"[ApiSubmit] negative prompt injected into node {negative_prompt_id}")
    if seed != -1:
        _inject_seed(wf, seed_id, seed)
        logger.info(f"[ApiSubmit] seed {seed} injected into node {seed_id}")

    if image is not None:
        if not image_node_id:
            raise ValueError("`image_node_id` is required when `image` is provided")
        png = _tensor_to_png_bytes(image)
        filename = f"api_input_{uuid.uuid4().hex}.png"
        uploaded = client.upload_image(png, filename)
        _inject_image(wf, image_node_id, uploaded)
        logger.info(f"[ApiSubmit] uploaded image as {uploaded!r}, injected into node {image_node_id}")

    _apply_overrides(wf, overrides)
    return client.submit(wf)


def _extract_images_tensor(
    client: ApiComfyClient, outputs: dict, output_node_id: str | None
) -> torch.Tensor:
    target = None
    if output_node_id and output_node_id in outputs:
        target = outputs[output_node_id]
    else:
        for _nid, out in outputs.items():
            if "images" in out or "gifs" in out:
                target = out
                break
    if target is None:
        raise RuntimeError("No image/gif outputs in remote response")

    entries = target.get("images") or target.get("gifs") or []

    def _fetch(img):
        return img, client.fetch_image(
            img["filename"], img.get("subfolder", ""), img.get("type", "output")
        )

    tensors = []
    workers = max(1, min(_FETCH_CONCURRENCY, len(entries)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_fetch, entries))
    for img, data in results:
        try:
            pil = Image.open(io.BytesIO(data))
        except Exception as e:
            # Non-image (e.g. mp4 from VHS_VideoCombine) — skip.
            logger.debug(f"[ApiGenerate] skipping non-image output {img.get('filename')}: {e}")
            continue
        if getattr(pil, "is_animated", False):
            for frame_idx in range(pil.n_frames):
                pil.seek(frame_idx)
                arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(arr))
        else:
            arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))

    if not tensors:
        raise RuntimeError("No decodable images extracted from remote outputs")
    return torch.stack(tensors, dim=0)


#################################################################
# Base class
#################################################################
class BaseApi:
    """Base class for API nodes."""



#################################################################
# Nodes
#################################################################
class ApiGenerate(BaseApi):
    """Send a workflow (API format JSON) to a remote ComfyUI instance and
    return the resulting image tensor (or any output, via wildcard).

    `workflow` must be the **API format** JSON (ComfyUI menu:
    "Save (API Format)"), not the UI workflow format.

    Args:
        workflow (str): API-format workflow JSON string, or path to a file.
        positive_prompt (str): Positive prompt text, injected into `positive_prompt_id`.
        positive_prompt_id (str): Node id receiving the positive prompt.
        negative_prompt_id (str): Node id receiving `negative_prompt` (when provided).
        output_id (str): Node id whose output (`images`/`gifs`) is fetched and decoded.
        seed (int): `-1` keeps the workflow's existing seed value.
        seed_id (str): Node id whose `seed`/`noise_seed` input receives `seed`.
        api_url (str): Remote ComfyUI base URL, e.g. `https://xxxx-8188.proxy.runpod.net`
            or `http://127.0.0.1:8188`.
        image_node_id (str): LoadImage node id receiving the uploaded `image`.
        timeout_sec (int): Max polling time in seconds.
        negative_prompt (str): Optional. Skipped if empty.
        image (IMAGE): Optional. Uploaded to remote and bound to `image_node_id`.
        overrides (str): Optional JSON object of the form
            `{node_id: <full node dict>}`. Each entry **replaces the entire node
            entry** in the workflow (not just `inputs` — provide `class_type`,
            `inputs`, etc.). Applied last, after prompt/seed/image injection,
            so it has the final word. Empty string disables.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow": ("STRING", {"forceInput": True}),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "positive_prompt_id": ("STRING", {}),
                "negative_prompt_id": ("STRING", {}),
                "output_id": ("STRING", {}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "seed_id": ("STRING", {}),
                "api_url": ("STRING", {}),
                "image_node_id": ("STRING", {}),
                "timeout_sec": ("INT", {"default": 300, "min": 1, "max": 36000}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"forceInput": True}),
                "image": ("IMAGE",),
                "overrides": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("output",)
    FUNCTION = "generate"
    CATEGORY = "AlcheminePack/API"

    @classmethod
    def IS_CHANGED(
        cls,
        workflow,
        positive_prompt,
        positive_prompt_id,
        negative_prompt_id,
        output_id,
        seed,
        seed_id,
        api_url,
        image_node_id,
        timeout_sec,
        negative_prompt="",
        image=None,
        overrides="",
    ):
        h = hashlib.sha256()
        for v in (
            workflow,
            positive_prompt,
            positive_prompt_id,
            negative_prompt_id,
            output_id,
            seed,
            seed_id,
            api_url,
            image_node_id,
            timeout_sec,
            negative_prompt,
            overrides,
        ):
            h.update(repr(v).encode("utf-8"))
            h.update(b"\x00")
        if image is not None:
            h.update(image.numpy().tobytes())
        return h.hexdigest()

    async def generate(
        self,
        workflow: str,
        positive_prompt: str,
        positive_prompt_id: str,
        negative_prompt_id: str,
        output_id: str,
        seed: int,
        seed_id: str,
        api_url: str,
        image_node_id: str,
        timeout_sec: int,
        negative_prompt: str = "",
        image: torch.Tensor | None = None,
        overrides: str = "",
    ) -> tuple[torch.Tensor]:
        # Offload the blocking submit/poll/fetch to a worker thread so the event
        # loop stays free. ComfyUI marks this node PENDING and runs other ready
        # nodes meanwhile, resuming downstream once this completes.
        return await asyncio.to_thread(
            self._generate_sync,
            workflow,
            positive_prompt,
            positive_prompt_id,
            negative_prompt_id,
            output_id,
            seed,
            seed_id,
            api_url,
            image_node_id,
            timeout_sec,
            negative_prompt,
            image,
            overrides,
        )

    def _generate_sync(
        self,
        workflow: str,
        positive_prompt: str,
        positive_prompt_id: str,
        negative_prompt_id: str,
        output_id: str,
        seed: int,
        seed_id: str,
        api_url: str,
        image_node_id: str,
        timeout_sec: int,
        negative_prompt: str = "",
        image: torch.Tensor | None = None,
        overrides: str = "",
    ) -> tuple[torch.Tensor]:
        _validate_api_url(api_url)

        wf = _load_workflow(workflow)
        _inject_text(wf, positive_prompt_id, positive_prompt)
        if negative_prompt:
            if not negative_prompt_id:
                raise ValueError("`negative_prompt_id` is required when `negative_prompt` is provided")
            _inject_text(wf, negative_prompt_id, negative_prompt)
            logger.info(f"[ApiGenerate] negative prompt injected into node {negative_prompt_id}")
        if seed != -1:
            _inject_seed(wf, seed_id, seed)
            logger.info(f"[ApiGenerate] seed {seed} injected into node {seed_id}")

        client = ApiComfyClient(api_url, None)

        if image is not None:
            if not image_node_id:
                raise ValueError("`image_node_id` is required when `image` is provided")
            png = _tensor_to_png_bytes(image)
            filename = f"api_input_{uuid.uuid4().hex}.png"
            uploaded = client.upload_image(png, filename)
            _inject_image(wf, image_node_id, uploaded)
            logger.info(f"[ApiGenerate] uploaded image as {uploaded!r}, injected into node {image_node_id}")

        _apply_overrides(wf, overrides)

        prompt_id = client.submit(wf)
        logger.info(f"[ApiGenerate] submitted prompt_id={prompt_id}, polling up to {timeout_sec}s...")
        outputs = client.wait(prompt_id, timeout=timeout_sec)
        image_tensor = _extract_images_tensor(client, outputs, output_id or None)
        logger.info(f"[ApiGenerate] received {image_tensor.shape[0]} frame(s)")
        return (image_tensor,)


class ApiSubmit(BaseApi):
    """Fire-and-forget submit to a remote ComfyUI instance.

    Builds the API workflow (same injection logic as `ApiGenerate`), submits
    it, records the in-flight job in `jobs.lock`, and returns immediately with
    the `job_id` — it does NOT wait for the remote to finish. This lets the
    local prompt complete right away so the next image can be queued. Collect the
    finished result later with `ApiCollect`.

    This is an OUTPUT_NODE so it runs even when nothing consumes its output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow": ("STRING", {"forceInput": True}),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "positive_prompt_id": ("STRING", {}),
                "negative_prompt_id": ("STRING", {}),
                "output_id": ("STRING", {}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "seed_id": ("STRING", {}),
                "api_url": ("STRING", {}),
                "image_node_id": ("STRING", {}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"forceInput": True}),
                "image": ("IMAGE",),
                "overrides": ("STRING", {"forceInput": True}),
                "label": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("job_id",)
    FUNCTION = "submit"
    OUTPUT_NODE = True
    CATEGORY = "AlcheminePack/API"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-submit on each queue (fire-and-forget per run).
        return time.time()

    async def submit(
        self,
        workflow: str,
        positive_prompt: str,
        positive_prompt_id: str,
        negative_prompt_id: str,
        output_id: str,
        seed: int,
        seed_id: str,
        api_url: str,
        image_node_id: str,
        negative_prompt: str = "",
        image: torch.Tensor | None = None,
        overrides: str = "",
        label: str = "",
    ) -> tuple[str]:
        return await asyncio.to_thread(
            self._submit_sync,
            workflow,
            positive_prompt,
            positive_prompt_id,
            negative_prompt_id,
            output_id,
            seed,
            seed_id,
            api_url,
            image_node_id,
            negative_prompt,
            image,
            overrides,
            label,
        )

    def _submit_sync(
        self,
        workflow,
        positive_prompt,
        positive_prompt_id,
        negative_prompt_id,
        output_id,
        seed,
        seed_id,
        api_url,
        image_node_id,
        negative_prompt,
        image,
        overrides,
        label,
    ) -> tuple[str]:
        _validate_api_url(api_url)
        # No queue: only one job in flight at a time. Guard the whole
        # check-then-submit so two concurrent submits can't both pass the
        # "is it free?" test.
        with _lock_guard:
            cur = _read_lock()
            if cur is not None:
                age = time.time() - cur.get("submitted_at", 0)
                if age < _STALE_SEC:
                    logger.debug(
                        f"[ApiSubmit] job already in progress "
                        f"(prompt_id={cur.get('prompt_id')}); skipping submit"
                    )
                    return {"ui": {"text": ["(skipped: in progress)"]}, "result": ("",)}
                logger.info(
                    f"[ApiSubmit] stale lock ({age:.0f}s old); overriding and resubmitting"
                )

            client = ApiComfyClient(api_url, None)
            prompt_id = _prepare_and_submit(
                client,
                workflow,
                positive_prompt,
                positive_prompt_id,
                negative_prompt_id,
                seed,
                seed_id,
                image_node_id,
                negative_prompt,
                image,
                overrides,
            )
            job_id = uuid.uuid4().hex
            _write_lock(
                {
                    "job_id": job_id,
                    "prompt_id": prompt_id,
                    "api_url": api_url.strip(),
                    "output_id": output_id,
                    "label": label,
                    "submitted_at": time.time(),
                }
            )
        logger.info(
            f"[ApiSubmit] submitted job {job_id} (prompt_id={prompt_id}, "
            f"label={label!r}); returning immediately"
        )
        return {"ui": {"text": [job_id]}, "result": (job_id,)}


class ApiCollect(BaseApi):
    """Collect the in-flight remote job submitted via `ApiSubmit`.

    Reads the `jobs.lock` file, polls the recorded job's `/history`, and when it
    is complete fetches the output, clears the lock (freeing the slot for the
    next submit), and returns the image/frame tensor — wire this into the
    existing post-processing graph (RIFE -> VideoCombine).

    Outcomes:
      - completed  -> return frames, clear lock
      - errored    -> drop job, clear lock, skip downstream
      - running    -> wait up to `wait_sec`, then skip downstream this run
      - no job     -> skip downstream

    "Skip" means returning an `ExecutionBlocker` so nothing downstream runs.
    Run this in a loop (e.g. `/loop`) to pick up the video once it finishes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 0 = don't wait: collect if ready, else skip immediately.
                "wait_sec": ("INT", {"default": 0, "min": 0, "max": 36000}),
                "poll_interval": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 60.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("output", "label")
    FUNCTION = "collect"
    CATEGORY = "AlcheminePack/API"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def _try_collect(self):
        """Poll the locked job once. Returns:
          (tensor, label) -> completed (lock cleared)
          "skip"          -> still running / no job
        Errored jobs clear the lock and return "skip"."""
        rec = _read_lock()
        if rec is None:
            return "skip"
        client = ApiComfyClient(rec["api_url"], None)
        try:
            r = client.session.get(
                f"{client.base}/history/{rec['prompt_id']}",
                headers=client.headers,
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            logger.warning(f"[ApiCollect] poll failed for {rec['prompt_id']}: {e}")
            return "skip"
        entry = data.get(rec["prompt_id"])
        if not entry:
            return "skip"  # not in history yet -> still running
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            logger.warning(f"[ApiCollect] job {rec['prompt_id']} errored on remote; clearing lock")
            with _lock_guard:
                _write_lock(None)
            return "skip"
        if not status.get("completed"):
            return "skip"
        tensor = _extract_images_tensor(client, entry["outputs"], rec.get("output_id") or None)
        with _lock_guard:
            _write_lock(None)
        logger.info(
            f"[ApiCollect] collected job {rec['prompt_id']} "
            f"({tensor.shape[0]} frame(s), label={rec.get('label')!r}); lock freed"
        )
        return tensor, rec.get("label", "")

    async def collect(self, wait_sec: int, poll_interval: float):
        return await asyncio.to_thread(self._collect_sync, wait_sec, poll_interval)

    def _collect_sync(self, wait_sec: int, poll_interval: float):
        deadline = time.time() + wait_sec
        while True:
            result = self._try_collect()
            if result != "skip":
                tensor, label = result
                return (tensor, label)
            if time.time() >= deadline:
                logger.debug("[ApiCollect] nothing to collect; skipping downstream")
                blocker = ExecutionBlocker(None)
                return (blocker, blocker)
            time.sleep(poll_interval)


class LoadWorkflow(BaseApi):
    """Load a workflow JSON file from `ComfyUI/user/default/workflows/`
    and return its contents as a STRING (API-format JSON).

    Pair with `ApiGenerate`'s `workflow_json` socket.
    """

    @classmethod
    def INPUT_TYPES(cls):
        root = _workflows_dir()
        files: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(".json"):
                    rel = os.path.relpath(os.path.join(dirpath, name), root)
                    files.append(rel.replace("\\", "/"))
        files.sort()
        return {
            "required": {
                "filename": (files,),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load"
    CATEGORY = "AlcheminePack/API"

    @classmethod
    def IS_CHANGED(cls, filename: str):
        path = os.path.join(_workflows_dir(), filename)
        try:
            return os.path.getmtime(path)
        except OSError:
            return float("nan")

    def load(self, filename: str) -> tuple[str]:
        path = os.path.join(_workflows_dir(), filename)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return (content,)
