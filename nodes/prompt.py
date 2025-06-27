import re
import logging
from functools import wraps


logger = logging.getLogger(__name__)


def exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.error(f"Unexpected error in {func.__name__}", exc_info=True)
            raise

    return wrapper


class ProcessTags:
    """Process tags from a prompt."""

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
            "blacklist": ("STRING", {"forceInput": True}),
            "remove_subtags": ("BOOLEAN", {"default": True}),
            "remove_tags": ("BOOLEAN", {"default": True}),
            "remove_underscores": ("BOOLEAN", {"default": True}),
        }
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_text", "removed_tags")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @exception_handler
    def execute(
        self,
        text: str,
        blacklist: str,
        remove_subtags: bool = True,
        remove_tags: bool = True,
        remove_underscores: bool = True,
    ):
        removed_tags_list = []
        if remove_tags:
            text, cur_removed_tags = RemoveTags().execute(text, blacklist)
            if cur_removed_tags:
                removed_tags_list.append(cur_removed_tags)
        if remove_subtags:
            text, cur_removed_tags = RemoveSubtags().execute(text)
            if cur_removed_tags:
                removed_tags_list.append(cur_removed_tags)
        if remove_underscores:
            text = RemoveUnderscores().execute(text)[0]

        return (text, removed_tags_list)


class RemoveTags:
    """Remove tags from a prompt."""

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
            "blacklist": ("STRING", {"forceInput": True}),
        }
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_text", "removed_tags")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @exception_handler
    def execute(self, text: str, blacklist: str):
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

        # 1. Split tokens by BREAK
        groups = text.split("BREAK")

        # 2. Compile blacklist
        compiled_blacklist = re.compile(
            r"|".join([t.strip() for t in blacklist.split(",")])
        )

        # 3. Remove all tags from blacklist from each group
        removed_tags = []
        new_groups = []
        for group in groups:
            # Ignore empty tags
            original_tags = [t for t in group.split(",") if t.strip()]
            comp_tags = [(idx, normalize_tag(t)) for idx, t in enumerate(original_tags)]
            valid_idxs = []
            for idx, tag in comp_tags:
                if not compiled_blacklist.search(tag):
                    valid_idxs.append(idx)
            new_group = ",".join([original_tags[idx] for idx in sorted(valid_idxs)])
            new_groups.append(new_group)
            removed_tags.extend(
                [
                    original_tags[idx].strip()
                    for idx in range(len(original_tags))
                    if idx not in valid_idxs
                ]
            )

        # 3. Join groups by BREAK
        new_text = "BREAK".join(new_groups)
        removed_tags = ", ".join(removed_tags)
        return (new_text, removed_tags)


class RemoveSubtags:
    """Remove all subtags from a prompt.

    Examples:
        Input: dog, cat, white dog, black cat
        Output: white dog, black cat

        Input: (cat:0.9), (cat:1.1), black cat, (black cat)
        Output: (cat:0.9), (cat:1.1), black cat, (black cat)
    """

    INPUT_TYPES = lambda: {"required": {"text": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_text", "removed_tags")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @exception_handler
    def execute(self, text: str):
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

        # 1. Split tokens by BREAK
        groups = text.split("BREAK")

        # 2. Remove all subtags from each group
        removed_tags = []
        new_groups = []
        for group in groups:
            # Ignore empty tags
            original_tags = [t for t in group.split(",") if t.strip()]
            comp_tags = [(idx, normalize_tag(t)) for idx, t in enumerate(original_tags)]
            valid_idxs = set()
            for idx, tag in sorted(
                comp_tags, key=lambda x: (len(x[1]), -x[0]), reverse=True
            ):
                if not any(tag in comp_tags[valid_idx][1] for valid_idx in valid_idxs):
                    valid_idxs.add(idx)
            new_group = ",".join([original_tags[idx] for idx in sorted(valid_idxs)])
            new_groups.append(new_group)
            removed_tags.extend(
                [
                    original_tags[idx].strip()
                    for idx in range(len(original_tags))
                    if idx not in valid_idxs
                ]
            )

        # 3. Join groups by BREAK
        new_text = "BREAK".join(new_groups)
        removed_tags = ", ".join(removed_tags)
        return (new_text, removed_tags)


class RemoveUnderscores:
    """Remove all underscores from a prompt.

    Examples:
        Input: dog_cat_white_dog_black_cat
        Output: dogcatwhitedogblackcat
    """

    INPUT_TYPES = lambda: {"required": {"text": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @exception_handler
    def execute(self, text: str):
        new_text = text.replace("_", " ")
        return (new_text,)
