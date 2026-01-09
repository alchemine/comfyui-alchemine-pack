"""Nodes in AlcheminePack/Danbooru."""

import os
import re
import asyncio
from math import ceil
from pathlib import Path
from random import Random
from collections import defaultdict
from os.path import exists, relpath, splitext

import folder_paths
from playwright.async_api import async_playwright

from .lib.utils import get_logger


logger = get_logger()


#################################################################
# Base class
#################################################################
class BaseDanbooru:
    """Base class for Danbooru nodes."""

    REQUEST_CACHE = {}

    @staticmethod
    def normalize_tag(tag: str) -> str:
        """Normalize tag with 2 decimal places.

        Examples:
            Input: cat
            Output: (cat:1.00)

            Input: (cat:1.2)
            Output: (cat:1.20)

            Input: ((cat))
            Output: (cat:1.21)

            Input: [cat]
            Output: (cat:0.90)

            Input: [[cat]]
            Output: (cat:0.81)

            Input: (cat:1.2:1.3)
            Output: (cat:1.20:1.30)
        """
        tag = tag.strip()
        if match := re.search(r"^\(([^()]+):([-0-9. ]+)\)$", tag):
            # Example: (cat:1.20)
            tag, weight = match.groups()
        elif match := re.search(r"^\(([^()]+):([0-9. ]+):([0-9. ]+)\)$", tag):
            # Example: (cat:1.20:1.30)
            tag, weight_s, weight_e = match.groups()
        elif re.match(r"^[^\(\[]", tag):
            # Example: cat
            pass
        elif match := re.search(r"^(\(+)(.+)(\)+)$", tag):
            # Example: (cat), ((cat))
            tag = match.group(2)
        elif match := re.search(r"^(\[+)(.+)(\]+)$", tag):
            # Example: [cat], [[cat]]
            tag = match.group(2)
        else:
            # logger.warning(f"Unexpected tag format: {tag}")
            pass
        return tag

    @staticmethod
    def remove_weight(tag: str) -> str:
        """Remove weight from a tag.

        Examples:
            Input: (cat:1.20)
            Output: cat
        """
        tag = tag.strip()

        if match := re.search(r"^\(([^()]+):[0-9.-]+\)$", tag):
            # Example: (cat:1.20)
            tag = match.group(1)
        elif match := re.search(r"^\(([^()]+):[0-9.-]+:[0-9.-]+\)$", tag):
            # Example: (cat:1.20:1.30)
            tag = match.group(1)
        elif match := re.search(r"^([\(\[]+)(.+)([\)\]]+)$", tag):
            # Example: (cat), ((cat)), [cat], [[cat]]
            tag = match.group(2)
        else:
            pass
        return tag

    @staticmethod
    def convert_to_danbooru_tag(tag: str) -> str:
        """Convert a tag to a Danbooru tag.

        Examples:
            Input: cat
            Output: cat
        """
        # 1. Replace spaces with underscores
        tag = tag.strip()
        tag = tag.replace(" ", "_")

        # 2. Replace parentheses with brackets
        tag = tag.replace(r"\(", r"(").replace(r"\)", r")")
        return tag

    @staticmethod
    def convert_from_danbooru_tag(tag: str) -> str:
        """Convert a Danbooru tag to a tag."""
        # 1. Replace parentheses with brackets
        tag = tag.strip()
        tag = tag.replace(r"(", r"\(").replace(r")", r"\)")

        # 2. Replace underscores with spaces
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
    async def aexecute(
        cls,
        text: str,
        category: str = "General",
        order: str = "Frequency",
        threshold: float = 0.5,
        n_min_tags: int = 0,
        n_max_tags: int = 100,
    ) -> tuple[str]:
        """Retrieve related tags by frequency from Danbooru."""
        queries = []
        groups = text.split("BREAK")
        for group in groups:
            for tag in group.split(","):
                tag = cls.remove_weight(tag)
                danbooru_tag = cls.convert_to_danbooru_tag(tag)
                queries.append(danbooru_tag)

        result_tags = []
        datas = await cls.arequest(queries, category, order)
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
    async def arequest(
        cls, queries: list[str], category: str, order: str
    ) -> list[dict]:
        """Request Danbooru API."""
        base_url = "https://danbooru.donmai.us/related_tag.json?commit=Search&search[category]={category}&search[order]={order}&search[query]={query}"

        responses = []
        async with async_playwright() as p:
            api_context = await p.request.new_context()
            for query in queries:
                url = base_url.format(category=category, order=order, query=query)
                if url in cls.REQUEST_CACHE:
                    responses.append(cls.REQUEST_CACHE[url])
                    continue

                resp = await api_context.get(url)
                if not resp.ok:
                    text = await resp.text()
                    logger.error(
                        f"Request to {url} failed with status {resp.status}: {text}"
                    )
                    raise Exception(
                        f"Request to {url} failed with status {resp.status}"
                    )
                json_data = await resp.json()
                cls.REQUEST_CACHE[url] = json_data
                responses.append(json_data)
        return responses

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
        return asyncio.run(
            cls.aexecute(
                text=text,
                category=category,
                order=order,
                threshold=threshold,
                n_min_tags=n_min_tags,
                n_max_tags=n_max_tags,
            )
        )

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
    async def aexecute(cls, post_id: str) -> tuple[str, str, str, str, str, str, str]:
        url = f"https://danbooru.donmai.us/posts/{post_id}.json"
        if url not in cls.REQUEST_CACHE:
            async with async_playwright() as p:
                api_context = await p.request.new_context()
                resp = await api_context.get(url)
                if not resp.ok:
                    text = await resp.text()
                    logger.error(
                        f"Request to {url} failed with status {resp.status}: {text}"
                    )
                    raise Exception(
                        f"Request to {url} failed with status {resp.status}"
                    )
                cls.REQUEST_CACHE[url] = await resp.json()
        data = cls.REQUEST_CACHE[url]

        # tag_string
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
    def execute(cls, post_id: str) -> tuple[str, str, str, str, str, str, str]:
        return asyncio.run(cls.aexecute(post_id=post_id))

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
    ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        return asyncio.run(
            cls.aexecute(date=date, scale=scale, n=n, random=random, seed=seed)
        )

    @classmethod
    async def aexecute(
        cls,
        date: str = "",
        scale: str = "day",
        n: int = 1,
        random: bool = True,
        seed: int = 0,
    ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        # Set default values
        datas = await cls.arequest(date, scale, n, random)
        if random:
            datas = Random(seed).sample(datas, n)
        else:
            datas = sorted(datas, key=lambda x: x["score"], reverse=True)
            datas = datas[:n]

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
    async def arequest(cls, date: str, scale: str, n: int, random: bool) -> list[dict]:
        """Request Danbooru API."""
        params = {}
        if date:
            params["date"] = date
        if scale:
            params["scale"] = scale

        datas = []
        async with async_playwright() as p:
            api_context = await p.request.new_context()
            if random:
                n_pages = n
            else:
                n_pages = ceil(n / cls.N_POSTS_PER_POPULAR_PAGE)
            for page in range(1, 1 + n_pages):
                params["page"] = page
                params_str = (
                    "?" + "&".join([f"{k}={v}" for k, v in params.items()])
                    if params
                    else ""
                )
                url = (
                    f"https://danbooru.donmai.us/explore/posts/popular.json{params_str}"
                )

                # Cache requests (avoid Too Many Requests errors)
                if url in cls.REQUEST_CACHE:
                    datas.extend(cls.REQUEST_CACHE[url])
                    continue

                resp = await api_context.get(url)
                if not resp.ok:
                    text = await resp.text()
                    logger.error(
                        f"Request to {url} failed with status {resp.status}: {text}"
                    )
                    raise Exception(
                        f"Request to {url} failed with status {resp.status}"
                    )
                json_data = await resp.json()
                cls.REQUEST_CACHE[url] = json_data
                datas.extend(json_data)
        return datas

    @classmethod
    def IS_CHANGED(
        cls, date: str, scale: str, n: int, random: bool, seed: int
    ) -> tuple:
        if random:
            return (date, scale, n, random, seed)
        else:
            return (date, scale, n)


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
        return asyncio.run(
            cls.aexecute(tags=tags, n=n, dir_path=dir_path, prefix=prefix)
        )

    @classmethod
    async def aexecute(
        cls,
        tags: str = "",
        n: int = 1,
        dir_path: str = "",
        prefix: str = "",
    ) -> tuple[list[str]]:
        # Set default values
        output_dir = Path(folder_paths.get_output_directory())
        dir_path_obj = output_dir / dir_path
        if not exists(dir_path_obj):
            os.makedirs(dir_path_obj, exist_ok=True)

        datas = await cls.arequest(tags, n)

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

        async with async_playwright() as p:
            api_context = await p.request.new_context()

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
                        resp = await api_context.get(file_url)
                        if not resp.ok:
                            raise Exception(f"HTTP {resp.status}")
                        with open(file_path, "wb") as f:
                            f.write(await resp.body())
                        logger.info(f"Downloaded {file_url} to {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to download {file_url}: {e}")
                        continue

                file_paths.append(relpath(file_path, output_dir))
                idx += 1

        return (file_paths,)

    @classmethod
    async def arequest(cls, tags: str, n: int) -> list[dict]:
        """Request Danbooru API."""
        params = {}
        params["tags"] = tags

        n_pages = ceil(n / cls.N_POSTS_PER_PAGE)

        datas = []
        async with async_playwright() as p:
            api_context = await p.request.new_context()
            for page in range(1, 1 + n_pages):
                params["page"] = page
                params_str = "&".join([f"{k}={v}" for k, v in params.items()])
                url = f"https://danbooru.donmai.us/posts.json?{params_str}"

                # Cache requests (avoid Too Many Requests errors)
                if url in cls.REQUEST_CACHE:
                    datas.extend(cls.REQUEST_CACHE[url])
                    continue

                resp = await api_context.get(url)
                if not resp.ok:
                    text = await resp.text()
                    logger.error(
                        f"Request to {url} failed with status {resp.status}: {text}"
                    )
                    raise Exception(
                        f"Request to {url} failed with status {resp.status}"
                    )
                json_data = await resp.json()
                cls.REQUEST_CACHE[url] = json_data
                datas.extend(json_data)
        return datas[:n]

    @classmethod
    def IS_CHANGED(cls, tags: str, n: int, dir_path: str, prefix: str) -> tuple:
        return (tags, n, dir_path, prefix)


if __name__ == "__main__":
    # result = DanbooruRelatedTagsRetriever.execute(
    #     text=r"ray \(arknights\), amiya \(arknights\)",
    #     threshold=0.3,
    #     category="General",
    #     order="Frequency",
    #     n_min_tags=10,
    #     n_max_tags=100,
    # )
    result = DanbooruPostTagsRetriever.execute(post_id="9557805")
    # result = DanbooruPopularPostsTagsRetriever.execute(
    #     date="", scale="day", n=1, random=False, seed=0
    # )
    # result = DanbooruPostsDownloader.execute(tags="1girl solo", n=1, dir_path="output")
    print(result)
