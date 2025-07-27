import re
import logging
import textwrap
from math import ceil
from time import sleep
from pathlib import Path
from random import sample
from random import seed as random_seed
from functools import wraps
from collections import defaultdict

import aiohttp
import asyncio
import requests


#################################################################
# Logger setup
#################################################################
ROOT_DIR = Path(__file__).parent.parent
CUSTOM_NODES_DIR = ROOT_DIR.parent


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
class BasePrompt:
    """Base class for prompt processing.

    This class provides a base class for prompt processing."""

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
            logger.warning(f"Unexpected tag format: {tag}")
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
        """Convert a Danbooru tag to a tag.

        Examples:
            Input: cat
            Output: cat
        """
        # 1. Replace parentheses with brackets
        tag = tag.strip()
        tag = tag.replace(r"(", r"\(").replace(r")", r"\)")

        # 2. Replace underscores with spaces
        tag = tag.replace("_", " ")
        return tag


#################################################################
# Nodes
#################################################################
class DanbooruRelatedTagsRetriever(BasePrompt):
    """Retrieve related tags by frequency from Danbooru.

    Examples:
        Input: ray (arknights)
        Output: ray (arknights), animal ears, pantyhose
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
            "category": (
                "STRING",
                {
                    "default": "General",
                    "choices": ["General", "Character", "Copyright", "Artist", "Meta"],
                },
            ),
            "order": (
                "STRING",
                {
                    "default": "Frequency",
                    "choices": ["Cosine", "Jaccard", "Overlap", "Frequency"],
                },
            ),
            "threshold": ("FLOAT", {"default": 0.3}),
            "n_min_tags": ("INT", {"default": 0, "min": 0}),
            "n_max_tags": ("INT", {"default": 100, "min": 1}),
        }
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    async def async_execute(
        cls,
        text: str,
        category: str = "General",
        order: str = "Frequency",
        threshold: float = 0.5,
        n_min_tags: int = 0,
        n_max_tags: int = 100,
    ) -> tuple[str]:
        """Retrieve related tags by frequency from Danbooru."""
        base_url = "https://danbooru.donmai.us/related_tag.json?commit=Search&search[category]={category}&search[order]={order}&search[query]={query}"

        queries = []
        groups = text.split("BREAK")
        for group in groups:
            for tag in group.split(","):
                tag = cls.remove_weight(tag)
                danbooru_tag = cls.convert_to_danbooru_tag(tag)
                queries.append(danbooru_tag)

        async with aiohttp.ClientSession() as session:
            tasks = []
            for query in queries:
                url = base_url.format(category=category, order=order, query=query)
                tasks.append(session.get(url))
            responses = await asyncio.gather(*tasks)
            datas = []
            for resp in responses:
                resp.raise_for_status()
                datas.append(await resp.json())

        result_tags = []
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

        # 3. Remove duplicates while preserving order
        seen = set()
        ordered_unique_tags = []
        for tag in result_tags:
            if tag not in seen:
                seen.add(tag)
                ordered_unique_tags.append(tag)

        processed_text = ", ".join(ordered_unique_tags)
        return (processed_text,)

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
            cls.async_execute(
                text=text,
                category=category,
                order=order,
                threshold=threshold,
                n_min_tags=n_min_tags,
                n_max_tags=n_max_tags,
            )
        )

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        text: str,
        category: str = "General",
        order: str = "Frequency",
        threshold: float = 0.5,
        n_min_tags: int = 0,
        n_max_tags: int = 100,
    ) -> bool:
        if n_min_tags <= n_max_tags:
            return True
        else:
            logger.error("n_min_tags must be less than or equal to n_max_tags")
            return False


class DanbooruPostTagsRetriever(BasePrompt):
    """Retrieve tags from a Danbooru post.

    Examples:
        Input: 1234567890
        Output: ray (arknights), animal ears, pantyhose
    """

    INPUT_TYPES = lambda: {
        "required": {
            "post_id": ("STRING", {}),
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
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    def execute(cls, post_id: str) -> tuple[str, str, str, str, str, str]:
        url = f"https://danbooru.donmai.us/posts/{post_id}.json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # tag_string
        convert = lambda key: ", ".join(
            map(cls.convert_from_danbooru_tag, data[key].split())
        )
        general_tags = convert("tag_string_general")
        character_tags = convert("tag_string_character")
        copyright_tags = convert("tag_string_copyright")
        artist_tags = convert("tag_string_artist")
        meta_tags = convert("tag_string_meta")

        # NOTE: meta tags are excluded from full_tags
        # full_tags = convert("tag_string")
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
        )

    @classmethod
    def IS_CHANGED(cls, post_id: str) -> str:
        return post_id


class DanbooruPopularPostsTagsRetriever(BasePrompt):
    """Retrieve popular posts' tags from Danbooru.

    TODO: cache requests (Too Many Requests errors)
    TODO: check score

    Examples:
        Input: date="2025-01-01", scale="day", n=1, random=True, seed=0
        Output: ray (arknights), animal ears, pantyhose
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
    CATEGORY = "AlcheminePack/Prompt"

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
            cls.async_execute(date=date, scale=scale, n=n, random=random, seed=seed)
        )

    @classmethod
    async def async_execute(
        cls,
        date: str = "",
        scale: str = "day",
        n: int = 1,
        random: bool = True,
        seed: int = 0,
    ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        N_POSTS_PER_POPULAR_PAGE = 60

        params = {}
        if date:
            params["date"] = date
        if scale:
            params["scale"] = scale

        async with aiohttp.ClientSession() as session:
            tasks = []

            if random:
                # NOTE: Select `n` posts from `n` pages
                n_pages = n
            else:
                n_pages = ceil(n / N_POSTS_PER_POPULAR_PAGE)
            for page in range(n_pages):
                params["page"] = page
                params_str = (
                    "?" + "&".join([f"{k}={v}" for k, v in params.items()])
                    if params
                    else ""
                )
                url = (
                    f"https://danbooru.donmai.us/explore/posts/popular.json{params_str}"
                )
                tasks.append(session.get(url))
            responses = await asyncio.gather(*tasks)

            datas = []
            for resp in responses:
                resp.raise_for_status()
                datas.extend(await resp.json())

        if random:
            random_seed(seed)
            datas = sample(datas, n)
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
            # full_tags = convert(data, "tag_string")
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
    def IS_CHANGED(
        cls, date: str, scale: str, n: int, random: bool, seed: int
    ) -> tuple:
        if random:
            return (date, scale, n, random, seed)
        else:
            return (date, scale, n)


if __name__ == "__main__":
    # result = DanbooruRelatedTagsRetriever.execute(
    #     text=r"ray \(arknights\), amiya \(arknights\)",
    #     threshold=0.3,
    #     category="General",
    #     order="Frequency",
    #     n_min_tags=10,
    #     n_max_tags=100,
    # )
    # result = DanbooruPostTagsRetriever.execute(post_id="9557805")
    result = DanbooruPopularPostsTagsRetriever.execute(
        date="", scale="day", n=2, random=False, seed=0
    )
    print(result[0])
