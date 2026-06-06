"""Nodes in AlcheminePack/Grok — Grok Imagine I2V video generation."""

import asyncio
import base64
import io
import os
import time

import folder_paths
import numpy as np
import requests
import torch
from PIL import Image

from comfy_api.latest import InputImpl, io as comfy_io, ui as comfy_ui
from comfy_execution.graph import ExecutionBlocker

from .lib import joblock

# This module's slot in the shared `jobs.lock` (the API nodes use "api").
_KIND = "grok"

# If an in-flight job is older than this, treat the slot as free so a stuck
# request_id can't block submits forever. Grok I2V can take a few minutes, so
# this is generous.
_STALE_SEC = 3600

# 토큰·client_id는 명시적 입력으로 받거나, 비어 있으면 환경변수에서 읽는다 (코드에
# 직접 넣지 않기). 환경변수가 없어도 팩 로딩이 깨지지 않도록 실제 API 호출 시점에 읽는다.
class _Credentials:
    """Grok 인증 정보를 들고 다니며, 401 시 토큰을 갱신한다."""

    def __init__(self, access_token: str, refresh_token: str, client_id: str):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.client_id = client_id

    @classmethod
    def resolve(cls, access_token: str = "", refresh_token: str = "",
                client_id: str = "") -> "_Credentials":
        """명시적 입력을 우선 사용하고, 비어 있으면 환경변수로 채운다."""
        access_token = access_token or os.environ.get("GROK_ACCESS_TOKEN", "")
        refresh_token = refresh_token or os.environ.get("GROK_REFRESH_TOKEN", "")
        client_id = client_id or os.environ.get("GROK_CLIENT_ID", "")
        missing = [
            name for name, val in (
                ("GROK_ACCESS_TOKEN", access_token),
                ("GROK_REFRESH_TOKEN", refresh_token),
                ("GROK_CLIENT_ID", client_id),
            ) if not val
        ]
        if missing:
            raise RuntimeError(
                f"Missing Grok credentials: {', '.join(missing)}. "
                "Provide them as node inputs, or set the corresponding "
                "environment variables and restart ComfyUI."
            )
        return cls(access_token, refresh_token, client_id)

    def refresh(self) -> str:
        """refresh token으로 새 access token 발급."""
        r = requests.post(
            "https://auth.x.ai/oauth2/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": self.client_id,
            },
        )
        if not r.ok:
            raise RuntimeError(
                f"Grok token refresh failed ({r.status_code}): {r.text}\n"
                "Your refresh token or client_id is likely expired/revoked. "
                "Re-authenticate with x.ai and update GROK_ACCESS_TOKEN / "
                "GROK_REFRESH_TOKEN / GROK_CLIENT_ID (in .env or the node inputs)."
            )
        d = r.json()
        self.access_token = d["access_token"]
        self.refresh_token = d.get("refresh_token", self.refresh_token)
        return self.access_token


def _api(creds: "_Credentials", method: str, url: str, **kw) -> requests.Response:
    """401/403이면 토큰 갱신 후 1회 재시도하는 래퍼.

    x.ai는 access token이 만료되면 401뿐 아니라 403도 반환하므로 둘 다 갱신을
    시도한다. 갱신 후에도 403이면 토큰 만료가 아니라 권한/차단 문제다.
    """
    kw.setdefault("headers", {})["Authorization"] = f"Bearer {creds.access_token}"
    resp = requests.request(method, url, **kw)
    if resp.status_code in (401, 403):
        kw["headers"]["Authorization"] = f"Bearer {creds.refresh()}"
        resp = requests.request(method, url, **kw)
    return resp


def _to_image_url(image: str) -> str:
    """image 인자를 API가 받는 url 형태로 변환.
    - 로컬 파일 경로 → data URL(base64)
    - 이미 http(s):// 나 data: 면 그대로 사용
    """
    if image.startswith(("http://", "https://", "data:")):
        return image
    ext = os.path.splitext(image)[1].lstrip(".").lower() or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{b64}"


def _tensor_to_data_url(image: torch.Tensor) -> str:
    """ComfyUI IMAGE 텐서 `(N,H,W,3)`의 첫 프레임을 PNG data URL로 변환."""
    if image.ndim == 4:
        image = image[0]
    arr = (image.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def generate_video(
    image: str,
    prompt: str = "",
    *,
    duration: int = 5,
    resolution: str = "720p",
    model: str = "grok-imagine-video-1.5-preview",
    out_path: str = "out.mp4",
    poll_interval: int = 5,
    timeout: int = 600,
    access_token: str = "",
    refresh_token: str = "",
    client_id: str = "",
) -> str:
    """이미지 한 장으로 Grok Imagine I2V 영상을 만들어 파일로 저장하고, 그 경로를 반환.

    Args:
        image: 로컬 파일 경로, http(s) URL, 또는 data URL.
        prompt: 움직임/연출 설명 (비워도 됨).
        duration: 1~15초.
        resolution: "480p" 또는 "720p".
        model: 막히면 "grok-imagine-video"(구버전)로 교체.
        out_path: 저장할 mp4 경로.

    Returns:
        저장된 영상 파일 경로(str).

    Raises:
        requests.HTTPError: API가 4xx/5xx를 반환한 경우.
        TimeoutError: timeout 안에 생성이 끝나지 않은 경우.
        RuntimeError: 생성이 failed/expired로 끝난 경우.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "image": {"url": _to_image_url(image)},
        "duration": duration,
        "resolution": resolution,
    }
    creds = _Credentials.resolve(access_token, refresh_token, client_id)
    return _run_job(creds, body, out_path, poll_interval, timeout)


def _submit_job(creds: "_Credentials", body: dict) -> str:
    """body로 생성 요청만 보내고 request_id를 반환 (폴링하지 않음)."""
    start = _api(creds, "POST", "https://api.x.ai/v1/videos/generations",
                 headers={"Content-Type": "application/json"}, json=body)
    start.raise_for_status()
    return start.json()["request_id"]


def _poll_job(creds: "_Credentials", request_id: str) -> "str | None":
    """request_id 상태를 1회 확인. done이면 영상 url, 아직이면 None,
    failed/expired면 RuntimeError."""
    res = _api(creds, "GET", f"https://api.x.ai/v1/videos/{request_id}").json()
    status = res["status"]
    if status == "done":
        return res["video"]["url"]
    if status in ("failed", "expired"):
        raise RuntimeError(f"Generation failed: {res.get('error', status)}")
    return None


def _download(url: str, out_path: str) -> str:
    with open(out_path, "wb") as f:
        f.write(requests.get(url).content)
    return out_path


def _run_job(creds: "_Credentials", body: dict, out_path: str,
             poll_interval: int, timeout: int) -> str:
    """이미 image url이 채워진 body로 생성 요청 → 폴링 → 파일 저장."""
    request_id = _submit_job(creds, body)
    deadline = time.time() + timeout
    while True:
        if time.time() > deadline:
            raise TimeoutError(
                f"Video generation did not finish within {timeout}s "
                f"(request_id={request_id})."
            )
        url = _poll_job(creds, request_id)
        if url:
            return _download(url, out_path)
        time.sleep(poll_interval)


def _reserve_output(filename_prefix: str):
    """output 폴더 안에 filename_prefix 규칙대로 충돌 없는 mp4 경로를 예약.
    (예: output/grok/GrokVideo_00001_.mp4). subfolder는 미리보기에 쓴다.

    Returns: (out_path, file, subfolder)
    """
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory()
    )
    file = f"{filename}_{counter:05d}_.mp4"
    out_path = os.path.join(full_output_folder, file)
    return out_path, file, subfolder


def _video_ui(out_path: str, file: str, subfolder: str):
    """저장된 mp4로 네이티브 VIDEO 객체와 인라인 미리보기 dict를 만든다.

    Returns: (video, preview_dict)
    """
    video = InputImpl.VideoFromFile(out_path)
    preview = comfy_ui.PreviewVideo(
        [comfy_ui.SavedResult(file, subfolder, comfy_io.FolderType.output)]
    ).as_dict()
    return video, preview


def _build_body(image, prompt, duration, resolution, model) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "image": {"url": _tensor_to_data_url(image)},
        "duration": duration,
        "resolution": resolution,
    }


class GrokGenerate:
    """이미지 한 장(IMAGE 텐서)으로 Grok Imagine I2V 영상을 만들어
    ComfyUI output 폴더에 저장하고, 네이티브 VIDEO(소리 포함)를 반환한다.
    노드 자체에서 인라인 영상 미리보기(재생/소리)가 뜬다.

    토큰·client_id는 노드 입력으로 직접 넣을 수 있고, 비워 두면 환경변수
    GROK_ACCESS_TOKEN / GROK_REFRESH_TOKEN / GROK_CLIENT_ID에서 읽는다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 15}),
                "resolution": (["720p", "480p"],),
                "model": ("STRING", {"default": "grok-imagine-video-1.5-preview"}),
                "filename_prefix": ("STRING", {"default": "grok/GrokVideo"}),
            },
            "optional": {
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 60}),
                "timeout": ("INT", {"default": 600, "min": 30, "max": 3600}),
                "access_token": ("STRING", {"default": ""}),
                "refresh_token": ("STRING", {"default": ""}),
                "client_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "AlcheminePack/Grok"
    OUTPUT_NODE = True

    def generate(
        self,
        image,
        prompt,
        duration,
        resolution,
        model,
        filename_prefix,
        poll_interval=5,
        timeout=600,
        access_token="",
        refresh_token="",
        client_id="",
    ):
        out_path, file, subfolder = _reserve_output(filename_prefix)
        body = _build_body(image, prompt, duration, resolution, model)
        creds = _Credentials.resolve(access_token, refresh_token, client_id)
        path = _run_job(creds, body, out_path, poll_interval, timeout)
        print(f"[GrokGenerate] 영상 저장 완료: {path}")

        # 네이티브 VIDEO 객체 (소리 포함) + 노드에서 바로 인라인 미리보기
        video, preview = _video_ui(path, file, subfolder)
        return {"ui": preview, "result": (video,)}


class GrokSubmit:
    """Grok Imagine I2V 영상 생성을 fire-and-forget로 제출한다.

    `GrokGenerate`와 같은 입력으로 생성 요청만 보내고, 진행 중 잡을 공유
    `jobs.lock`의 "grok" 슬롯에 기록한 뒤 즉시 `request_id`를 반환한다 — 원격
    완료를 기다리지 않으므로 로컬 프롬프트가 바로 끝나 다음 작업을 큐에 넣을 수
    있다. 완성된 영상은 나중에 `GrokCollect`로 수거한다.

    OUTPUT_NODE라서 출력을 소비하는 노드가 없어도 실행된다. mp4 저장 경로는
    제출 시점에 예약되어 lock에 기록되고, 수거 시 그 경로에 저장된다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 15}),
                "resolution": (["720p", "480p"],),
                "model": ("STRING", {"default": "grok-imagine-video-1.5-preview"}),
                "filename_prefix": ("STRING", {"default": "grok/GrokVideo"}),
            },
            "optional": {
                "access_token": ("STRING", {"default": ""}),
                "refresh_token": ("STRING", {"default": ""}),
                "client_id": ("STRING", {"default": ""}),
                "label": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("request_id",)
    FUNCTION = "submit"
    CATEGORY = "AlcheminePack/Grok"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-submit on each queue (fire-and-forget per run).
        return time.time()

    async def submit(self, image, prompt, duration, resolution, model,
                     filename_prefix, access_token="", refresh_token="",
                     client_id="", label=""):
        return await asyncio.to_thread(
            self._submit_sync, image, prompt, duration, resolution, model,
            filename_prefix, access_token, refresh_token, client_id, label,
        )

    def _submit_sync(self, image, prompt, duration, resolution, model,
                     filename_prefix, access_token, refresh_token, client_id,
                     label):
        creds = _Credentials.resolve(access_token, refresh_token, client_id)
        # No queue: only one Grok job in flight at a time. Guard the whole
        # check-then-submit so two concurrent submits can't both pass.
        with joblock.guard:
            cur = joblock.read_lock(_KIND)
            if cur is not None:
                age = time.time() - cur.get("submitted_at", 0)
                if age < _STALE_SEC:
                    print(
                        f"[GrokSubmit] job already in progress "
                        f"(request_id={cur.get('request_id')}); skipping submit"
                    )
                    return {"ui": {"text": ["(skipped: in progress)"]}, "result": ("",)}
                print(f"[GrokSubmit] stale lock ({age:.0f}s old); overriding and resubmitting")

            out_path, file, subfolder = _reserve_output(filename_prefix)
            body = _build_body(image, prompt, duration, resolution, model)
            request_id = _submit_job(creds, body)
            joblock.write_lock(
                _KIND,
                {
                    "request_id": request_id,
                    "out_path": out_path,
                    "file": file,
                    "subfolder": subfolder,
                    "label": label,
                    "submitted_at": time.time(),
                },
            )
        print(
            f"[GrokSubmit] submitted request_id={request_id} (label={label!r}); "
            "returning immediately"
        )
        return {"ui": {"text": [request_id]}, "result": (request_id,)}


class GrokCollect:
    """`GrokSubmit`으로 제출한 진행 중 Grok 잡을 수거한다.

    공유 `jobs.lock`의 "grok" 슬롯을 읽어 해당 잡의 상태를 폴링하고, 완료되면
    제출 시 예약된 경로에 영상을 저장한 뒤 lock을 비우고(다음 제출 슬롯 해제)
    네이티브 VIDEO를 반환한다.

    토큰·client_id는 GET 폴링에 필요하므로 여기서도 입력으로 받거나, 비우면
    환경변수에서 읽는다 (토큰을 lock 파일에 저장하지 않는다).

    결과:
      - done     -> 영상 저장, lock 해제, VIDEO 반환
      - failed   -> 잡 폐기, lock 해제, 다운스트림 차단
      - running  -> `wait_sec`까지 대기 후 이번 실행은 다운스트림 차단
      - no job   -> 다운스트림 차단

    "차단"은 `ExecutionBlocker`를 반환해 다운스트림이 실행되지 않게 한다.
    `/loop` 등으로 반복 실행하면 완료된 영상을 받아올 수 있다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 0 = don't wait: collect if ready, else skip immediately.
                "wait_sec": ("INT", {"default": 0, "min": 0, "max": 3600}),
                "poll_interval": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 60.0}),
            },
            "optional": {
                "access_token": ("STRING", {"default": ""}),
                "refresh_token": ("STRING", {"default": ""}),
                "client_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "label")
    FUNCTION = "collect"
    CATEGORY = "AlcheminePack/Grok"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def _try_collect(self, creds: "_Credentials"):
        """Poll the locked Grok job once. Returns:
          (out_path, file, subfolder, label) -> done (lock cleared)
          "skip"                             -> still running / no job
        failed/expired jobs clear the lock and return "skip"."""
        rec = joblock.read_lock(_KIND)
        if rec is None:
            return "skip"
        try:
            url = _poll_job(creds, rec["request_id"])
        except RuntimeError as e:
            print(f"[GrokCollect] job {rec['request_id']} failed on remote; clearing lock: {e}")
            with joblock.guard:
                joblock.write_lock(_KIND, None)
            return "skip"
        except requests.RequestException as e:
            print(f"[GrokCollect] poll failed for {rec['request_id']}: {e}")
            return "skip"
        if not url:
            return "skip"  # still running
        _download(url, rec["out_path"])
        with joblock.guard:
            joblock.write_lock(_KIND, None)
        print(
            f"[GrokCollect] collected request_id={rec['request_id']} "
            f"-> {rec['out_path']} (label={rec.get('label')!r}); lock freed"
        )
        return rec["out_path"], rec["file"], rec["subfolder"], rec.get("label", "")

    async def collect(self, wait_sec, poll_interval, access_token="",
                      refresh_token="", client_id=""):
        return await asyncio.to_thread(
            self._collect_sync, wait_sec, poll_interval,
            access_token, refresh_token, client_id,
        )

    def _collect_sync(self, wait_sec, poll_interval, access_token,
                      refresh_token, client_id):
        creds = _Credentials.resolve(access_token, refresh_token, client_id)
        deadline = time.time() + wait_sec
        while True:
            result = self._try_collect(creds)
            if result != "skip":
                out_path, file, subfolder, label = result
                video, preview = _video_ui(out_path, file, subfolder)
                return {"ui": preview, "result": (video, label)}
            if time.time() >= deadline:
                print("[GrokCollect] nothing to collect; skipping downstream")
                blocker = ExecutionBlocker(None)
                return (blocker, blocker)
            time.sleep(poll_interval)


# 사용 예 (단독 실행용)
if __name__ == "__main__":
    path = generate_video("start.png", "카메라가 천천히 줌인하며 머리카락이 바람에 흩날린다")
    print("저장 완료:", path)
