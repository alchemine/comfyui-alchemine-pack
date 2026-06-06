"""Nodes in AlcheminePack/Danbooru — `requests`-based implementation.

A drop-in alternative to `danbooru.py` that talks to the Danbooru JSON API with
plain `requests` instead of Playwright. These nodes never drove a browser — they
only GET `.json` endpoints (optionally through a Webshare proxy) — so `requests`
does the same work without the `playwright install` browser dependency.

To use these instead of the Playwright version, swap the import in `__init__.py`:
    from .nodes.danbooru_requests import (
        DanbooruRelatedTagsRetriever, ...
    )
"""

import os
import re
import time
from os import environ
from math import ceil
from pathlib import Path
from random import Random
from collections import defaultdict
from os.path import exists, relpath, splitext

import folder_paths
import requests

from .lib.utils import get_logger


logger = get_logger()


# Cache TTL (seconds) for volatile endpoints — popular / related / search,
# whose response for the same URL changes over time. Immutable endpoints (a
# single post by id) pass no ttl and are cached for the process lifetime.
VOLATILE_TTL = 3600  # 1 hour


# A single pooled session shared across all Danbooru nodes. A User-Agent is set
# because Danbooru may reject requests without one. No retries: a single failed
# request raises immediately instead of hammering the endpoint (and danbooru's
# connection resets are intermittent network-level RSTs, which retrying won't fix).
_HEADERS = {"User-Agent": "comfyui-alchemine-pack/danbooru (requests)"}
_session: "requests.Session | None" = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        # Ignore ambient HTTP_PROXY/HTTPS_PROXY/.netrc from the environment: this
        # node manages proxying explicitly via `get_proxies()`. Otherwise requests
        # silently routes through a stray OS proxy env var (e.g. set in the shell
        # that launched ComfyUI), which can break a connection that works fine
        # from a clean shell.
        s.trust_env = False
        s.headers.update(_HEADERS)
        _session = s
    return _session


#################################################################
# Base class
#################################################################
class BaseDanbooru:
    """Base class for Danbooru nodes (requests-based)."""

    REQUEST_CACHE = {}

    @classmethod
    def get_proxies(cls) -> "dict | None":
        """Return a `requests` proxies dict from env, or None if not set.

        Why: a Webshare rotating proxy avoids Danbooru rate-limits / IP bans.
        Reads WEBSHARE_PROXY_USERNAME / WEBSHARE_PROXY_PASSWORD from .env;
        WEBSHARE_PROXY_SERVER is optional (defaults to the Webshare endpoint).
        The credentials are embedded into the proxy URL as `user:pass@host`.
        """
        username = environ.get("WEBSHARE_PROXY_USERNAME")
        password = environ.get("WEBSHARE_PROXY_PASSWORD")
        if not username or not password:
            return None
        server = environ.get("WEBSHARE_PROXY_SERVER", "http://p.webshare.io:80")
        scheme, _, host = server.partition("://")
        if not host:  # server given without a scheme
            scheme, host = "http", server
        proxy_url = f"{scheme}://{username}:{password}@{host}"
        return {"http": proxy_url, "https": proxy_url}

    @classmethod
    def _get_json(cls, url: str, ttl: "float | None" = None) -> dict | list:
        """GET a JSON endpoint with caching (avoids Too Many Requests errors).

        Cache entries are stored as ``(expires_at, data)``. ``ttl`` is the cache
        lifetime in seconds: ``None`` caches for the process lifetime (immutable
        endpoints, e.g. a single post by id), while a positive value re-fetches
        once the entry expires (volatile endpoints whose response for the same
        URL drifts over time, e.g. popular/related/search).
        """
        now = time.time()
        entry = cls.REQUEST_CACHE.get(url)
        if entry is not None:
            expires_at, data = entry
            if expires_at is None or now < expires_at:
                return data
        resp = _get_session().get(url, proxies=cls.get_proxies(), timeout=30)
        if not resp.ok:
            logger.error(
                f"Request to {url} failed with status {resp.status_code}: {resp.text}"
            )
            raise Exception(f"Request to {url} failed with status {resp.status_code}")
        data = resp.json()
        cls.REQUEST_CACHE[url] = (None if ttl is None else now + ttl, data)
        return data

    @staticmethod
    def normalize_tag(tag: str) -> str:
        """Normalize tag with 2 decimal places.

        Examples:
            Input: cat            -> (cat:1.00)
            Input: (cat:1.2)      -> (cat:1.20)
            Input: [cat]          -> (cat:0.90)
            Input: (cat:1.2:1.3)  -> (cat:1.20:1.30)
        """
        tag = tag.strip()
        if match := re.search(r"^\(([^()]+):([-0-9. ]+)\)$", tag):
            tag, weight = match.groups()
        elif match := re.search(r"^\(([^()]+):([0-9. ]+):([0-9. ]+)\)$", tag):
            tag, weight_s, weight_e = match.groups()
        elif re.match(r"^[^\(\[]", tag):
            pass
        elif match := re.search(r"^(\(+)(.+)(\)+)$", tag):
            tag = match.group(2)
        elif match := re.search(r"^(\[+)(.+)(\]+)$", tag):
            tag = match.group(2)
        else:
            pass
        return tag

    @staticmethod
    def remove_weight(tag: str) -> str:
        """Remove weight from a tag.

        Example: (cat:1.20) -> cat
        """
        tag = tag.strip()
        if match := re.search(r"^\(([^()]+):[0-9.-]+\)$", tag):
            tag = match.group(1)
        elif match := re.search(r"^\(([^()]+):[0-9.-]+:[0-9.-]+\)$", tag):
            tag = match.group(1)
        elif match := re.search(r"^([\(\[]+)(.+)([\)\]]+)$", tag):
            tag = match.group(2)
        else:
            pass
        return tag

    @staticmethod
    def convert_to_danbooru_tag(tag: str) -> str:
        """Convert a tag to a Danbooru tag (spaces->underscores, unescape parens)."""
        tag = tag.strip()
        tag = tag.replace(" ", "_")
        tag = tag.replace(r"\(", r"(").replace(r"\)", r")")
        return tag

    @staticmethod
    def convert_from_danbooru_tag(tag: str) -> str:
        """Convert a Danbooru tag to a tag (escape parens, underscores->spaces)."""
        tag = tag.strip()
        tag = tag.replace(r"(", r"\(").replace(r")", r"\)")
        tag = tag.replace("_", " ")
        return tag


#################################################################
# Nodes
#################################################################
class DanbooruRelatedTagsRetriever(BaseDanbooru):
    """Retrieve related tags by frequency from Danbooru.

    Examples:
        Input: ray (arknights)
        Output: ray (arknights), animal ears, pantyhose
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {}),
            "category": (
                ["General", "Character", "Copyright", "Artist", "Meta"],
                {"default": "General"},
            ),
            "order": (
                ["Cosine", "Jaccard", "Overlap", "Frequency"],
                {"default": "Frequency"},
            ),
            "threshold": ("FLOAT", {"default": 0.3}),
            "n_min_tags": ("INT", {"default": 0, "min": 0}),
            "n_max_tags": ("INT", {"default": 100, "min": 1}),
        }
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Danbooru"

    @classmethod
    def execute(
        cls,
        text: str,
        category: str = "General",
        order: str = "Frequency",
        threshold: float = 0.5,
        n_min_tags: int = 0,
        n_max_tags: int = 100,
    ) -> tuple[str]:
        queries = []
        groups = text.split("BREAK")
        for group in groups:
            for tag in group.split(","):
                tag = cls.remove_weight(tag)
                danbooru_tag = cls.convert_to_danbooru_tag(tag)
                queries.append(danbooru_tag)

        result_tags = []
        datas = cls.request(queries, category, order)
        for query, data in zip(queries, datas):
            order_map = {
                "Cosine": "cosine_similarity",
                "Jaccard": "jaccard_similarity",
                "Overlap": "overlap_coefficient",
                "Frequency": "frequency",
            }
            related_tags_cands = [
                t for t in data["related_tags"] if not t["tag"]["is_deprecated"]
            ]
            related_tags_selected = [
                t for t in related_tags_cands if t[order_map[order]] >= threshold
            ]
            if n_min_tags and len(related_tags_selected) < n_min_tags:
                related_tags_selected = related_tags_cands[:n_min_tags]
            if n_max_tags:
                related_tags_selected = related_tags_selected[:n_max_tags]
            related_tags_selected = [
                cls.convert_from_danbooru_tag(t["tag"]["name"])
                for t in related_tags_selected
            ]
            result_tags.append(cls.convert_from_danbooru_tag(query))
            result_tags.extend(related_tags_selected)

        # Remove duplicates while preserving order
        seen = set()
        ordered_unique_tags = []
        for tag in result_tags:
            if tag not in seen:
                seen.add(tag)
                ordered_unique_tags.append(tag)

        processed_text = ", ".join(ordered_unique_tags)
        return (processed_text,)

    @classmethod
    def request(cls, queries: list[str], category: str, order: str) -> list[dict]:
        """Request the Danbooru related_tag API for each query."""
        base_url = "https://danbooru.donmai.us/related_tag.json?commit=Search&search[category]={category}&search[order]={order}&search[query]={query}"
        responses = []
        for query in queries:
            url = base_url.format(category=category, order=order, query=query)
            responses.append(cls._get_json(url, ttl=VOLATILE_TTL))
        return responses

    @classmethod
    def IS_CHANGED(
        cls,
        text: str,
        category: str = "General",
        order: str = "Frequency",
        threshold: float = 0.5,
        n_min_tags: int = 0,
        n_max_tags: int = 100,
    ) -> tuple:
        return (text, category, order, threshold, n_min_tags, n_max_tags)


class DanbooruPostTagsRetriever(BaseDanbooru):
    """Retrieve tags from a Danbooru post.

    Examples:
        Input: 1
        Output: kousaka tamaki, ...

    NOTE: meta tags are excluded from full_tags
    """

    INPUT_TYPES = lambda: {
        "required": {
            "post_id": ("STRING",),
        }
    }
    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "full_tags",
        "general_tags",
        "character_tags",
        "copyright_tags",
        "artist_tags",
        "meta_tags",
        "image_url",
    )
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Danbooru"

    @classmethod
    def execute(cls, post_id: str) -> tuple[str, str, str, str, str, str, str]:
        url = f"https://danbooru.donmai.us/posts/{post_id}.json"
        data = cls._get_json(url)

        convert = lambda key: ", ".join(
            map(cls.convert_from_danbooru_tag, data[key].split())
        )
        general_tags = convert("tag_string_general")
        character_tags = convert("tag_string_character")
        copyright_tags = convert("tag_string_copyright")
        artist_tags = convert("tag_string_artist")
        meta_tags = convert("tag_string_meta")
        image_url = data.get("file_url", "not_found")

        # NOTE: meta tags are excluded from full_tags
        full_tags = ", ".join(
            [character_tags, copyright_tags, artist_tags, general_tags]
        )

        return (
            full_tags,
            general_tags,
            character_tags,
            copyright_tags,
            artist_tags,
            meta_tags,
            image_url,
        )

    @classmethod
    def IS_CHANGED(cls, post_id: str) -> str:
        return post_id


class DanbooruPopularPostsTagsRetriever(BaseDanbooru):
    """Retrieve popular posts' tags from Danbooru.

    Examples:
        Input: date="2025-01-01", scale="day", n=1, random=True, seed=0
        Output: ray (arknights), animal ears, pantyhose

    NOTE: meta tags are excluded from full_tags
    """

    INPUT_TYPES = lambda: {
        "required": {
            "date": ("STRING", {"default": ""}),
            "scale": (
                ["day", "week", "month"],
                {"default": "day"},
            ),
            "n": ("INT", {"default": 1, "min": 1}),
            "random": ("BOOLEAN", {"default": True}),
            "seed": ("INT", {"default": 0}),
            # Rank cursor for ordered mode (random=False): returns the posts at
            # ranks [offset, offset+n) in popularity order. control_after_generate
            # gives it the same increment/fixed dropdown as seed, so each queue
            # can step one rank down. Ignored when random=True.
            "offset": ("INT", {"default": 0, "min": 0, "control_after_generate": True}),
        }
    }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "full_tags",
        "general_tags",
        "character_tags",
        "copyright_tags",
        "artist_tags",
        "meta_tags",
    )
    OUTPUT_IS_LIST = (True, True, True, True, True, True)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Danbooru"

    N_POSTS_PER_POPULAR_PAGE = 20  # Basic level limit

    @classmethod
    def execute(
        cls,
        date: str = "",
        scale: str = "day",
        n: int = 1,
        random: bool = True,
        seed: int = 0,
        offset: int = 0,
    ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        if random:
            # Pool n pages of popular posts and randomly sample n of them.
            datas = cls._fetch_pages(date, scale, range(1, 1 + n))
            datas = Random(seed).sample(datas, n)
        else:
            # Ordered mode: the popular endpoint already returns posts in rank
            # order (PER_PAGE per page), so fetch only the page(s) covering ranks
            # [offset, offset+n) instead of everything up to offset.
            per = cls.N_POSTS_PER_POPULAR_PAGE
            first_page = offset // per + 1
            last_page = (offset + n - 1) // per + 1
            pages = cls._fetch_pages(date, scale, range(first_page, last_page + 1))
            base = (first_page - 1) * per  # rank of pages[0]
            datas = pages[offset - base : offset - base + n]
            if len(datas) < n:
                raise ValueError(
                    f"Popular posts at rank {offset}..{offset + n - 1} don't exist "
                    f"(scale={scale!r}, date={date or 'latest'!r}); only "
                    f"{base + len(pages)} popular posts available."
                )

        convert = lambda data, key: ", ".join(
            map(cls.convert_from_danbooru_tag, data[key].split())
        )

        result = defaultdict(list)
        for data in datas:
            general_tags = convert(data, "tag_string_general")
            character_tags = convert(data, "tag_string_character")
            copyright_tags = convert(data, "tag_string_copyright")
            artist_tags = convert(data, "tag_string_artist")
            meta_tags = convert(data, "tag_string_meta")

            # NOTE: meta tags are excluded from full_tags
            full_tags = ", ".join(
                [character_tags, copyright_tags, artist_tags, general_tags]
            )
            result["full_tags"].append(full_tags)
            result["general_tags"].append(general_tags)
            result["character_tags"].append(character_tags)
            result["copyright_tags"].append(copyright_tags)
            result["artist_tags"].append(artist_tags)
            result["meta_tags"].append(meta_tags)

        return (
            result["full_tags"],
            result["general_tags"],
            result["character_tags"],
            result["copyright_tags"],
            result["artist_tags"],
            result["meta_tags"],
        )

    @classmethod
    def _fetch_pages(cls, date: str, scale: str, pages) -> list[dict]:
        """Fetch the given 1-indexed popular pages, concatenated in rank order."""
        params = {}
        if date:
            params["date"] = date
        if scale:
            params["scale"] = scale

        datas = []
        for page in pages:
            params["page"] = page
            params_str = "?" + "&".join(f"{k}={v}" for k, v in params.items())
            url = f"https://danbooru.donmai.us/explore/posts/popular.json{params_str}"
            datas.extend(cls._get_json(url, ttl=VOLATILE_TTL))
        return datas

    @classmethod
    def IS_CHANGED(
        cls, date: str, scale: str, n: int, random: bool, seed: int, offset: int = 0
    ) -> tuple:
        if random:
            return (date, scale, n, random, seed)
        else:
            return (date, scale, n, offset)


class DanbooruPostsDownloader(BaseDanbooru):
    """Download posts from Danbooru."""

    N_POSTS_PER_PAGE = 20  # Danbooru API default limit

    INPUT_TYPES = lambda: {
        "required": {
            "tags": ("STRING", {"default": ""}),
            "n": ("INT", {"default": 1, "min": 1}),
            "dir_path": ("STRING", {"default": ""}),
            "prefix": ("STRING", {"default": ""}),
        }
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_paths",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Danbooru"

    @classmethod
    def execute(
        cls,
        tags: str = "",
        n: int = 1,
        dir_path: str = "",
        prefix: str = "",
    ) -> tuple[list[str]]:
        output_dir = Path(folder_paths.get_output_directory())
        dir_path_obj = output_dir / dir_path
        if not exists(dir_path_obj):
            os.makedirs(dir_path_obj, exist_ok=True)

        datas = cls.request(tags, n)

        if prefix:
            start_idx = 1 + len(list(dir_path_obj.glob(f"{prefix}_*.*")))
        else:
            idxs = set()
            for f in dir_path_obj.glob("[0-9]*.*"):
                try:
                    idx_val = int(f.stem.split("_")[0])
                    idxs.add(idx_val)
                except (ValueError, IndexError):
                    continue
            start_idx = (max(idxs) + 1) if idxs else 1
        idx = start_idx

        session = _get_session()
        proxies = cls.get_proxies()

        file_paths = []
        for data in datas:
            if not data.get("file_url"):
                continue

            file_url = data["file_url"]
            extension = splitext(file_url.split("?")[0])[-1]
            if prefix:
                file_name = f"{prefix}_{idx}{extension}"
            else:
                file_name = f"{idx}{extension}"
            file_path = dir_path_obj / file_name

            if not file_path.exists():
                try:
                    resp = session.get(file_url, proxies=proxies, timeout=120)
                    if not resp.ok:
                        raise Exception(f"HTTP {resp.status_code}")
                    with open(file_path, "wb") as f:
                        f.write(resp.content)
                    logger.info(f"Downloaded {file_url} to {file_path}")
                except Exception as e:
                    logger.error(f"Failed to download {file_url}: {e}")
                    continue

            file_paths.append(relpath(file_path, output_dir))
            idx += 1

        return (file_paths,)

    @classmethod
    def request(cls, tags: str, n: int) -> list[dict]:
        """Request the Danbooru posts API."""
        params = {"tags": tags}
        n_pages = ceil(n / cls.N_POSTS_PER_PAGE)

        datas = []
        for page in range(1, 1 + n_pages):
            params["page"] = page
            params_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"https://danbooru.donmai.us/posts.json?{params_str}"
            datas.extend(cls._get_json(url, ttl=VOLATILE_TTL))
        return datas[:n]

    @classmethod
    def IS_CHANGED(cls, tags: str, n: int, dir_path: str, prefix: str) -> tuple:
        return (tags, n, dir_path, prefix)


if __name__ == "__main__":
    result = DanbooruPostTagsRetriever.execute(post_id="9557805")
    print(result)
