import re
import textwrap
from functools import wraps

import yaml

from .utils import WILDCARD_PATH, get_logger, exception_handler


logger = get_logger()


#################################################################
# Utility functions
#################################################################
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


#################################################################
# Base class
#################################################################
class BasePrompt:
    """Base class for Prompt nodes."""

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

    @classmethod
    def adjust_fixed_tags(cls, text: str, fixed_tags: str) -> str:
        """Adjust fixed tags to be in the same order as tags in the text."""
        if fixed_tags:
            fixed_tags_set = set(
                [
                    cls.normalize_tag(t)
                    for t in re.split(r"BREAK|,", fixed_tags)
                    if t.strip()
                ]
            )
            tags_set = set(
                [cls.normalize_tag(t) for t in re.split(r"BREAK|,", text) if t.strip()]
            )
            added_texts = ", ".join(fixed_tags_set - tags_set)
            text = f"{text}, {added_texts}"

        return text


#################################################################
# Nodes
#################################################################
class ProcessTags(BasePrompt):
    """Full process of tags from a prompt.

    Order of operations: ReplaceUnderscores -> FilterTags -> FilterSubtags"""

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
            "replace_underscores": ("BOOLEAN", {"default": True}),
            "filter_tags": ("BOOLEAN", {"default": True}),
            "filter_subtags": ("BOOLEAN", {"default": True}),
        },
        "optional": {
            "blacklist_tags": ("STRING", {"default": ""}),
            "fixed_tags": ("STRING", {"default": ""}),
        },
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_text", "filtered_tags_list")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    def execute(
        cls,
        text: str,
        replace_underscores: bool = True,
        filter_tags: bool = True,
        filter_subtags: bool = True,
        blacklist_tags: str = "",
        fixed_tags: str = "",
    ) -> tuple[str, list[str]]:
        """Process tags from a prompt."""
        text = cls.adjust_fixed_tags(text, fixed_tags)

        filtered_tags_list = []

        if replace_underscores:
            text = ReplaceUnderscores.execute(text=text)[0]

        if filter_tags:
            text, cur_filtered_tags = FilterTags.execute(
                text=text, blacklist_tags=blacklist_tags, fixed_tags=fixed_tags
            )
            if cur_filtered_tags:
                filtered_tags_list.append(cur_filtered_tags)

        if filter_subtags:
            text, cur_filtered_tags = FilterSubtags.execute(
                text=text, fixed_tags=fixed_tags
            )
            if cur_filtered_tags:
                filtered_tags_list.append(cur_filtered_tags)

        return (text, filtered_tags_list)

    @classmethod
    def IS_CHANGED(
        cls,
        text: str,
        replace_underscores: bool = True,
        filter_tags: bool = True,
        filter_subtags: bool = True,
        blacklist_tags: str = "",
        fixed_tags: str = "",
    ) -> bool:
        return (
            text,
            replace_underscores,
            filter_tags,
            filter_subtags,
            blacklist_tags,
            fixed_tags,
        )


class FilterTags(BasePrompt):
    """Filter blacklisted tags from a prompt. Regular expression is used to match tags."""

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
        },
        "optional": {
            "blacklist_tags": ("STRING", {"default": ""}),
            "fixed_tags": ("STRING", {"default": ""}),
        },
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_text", "filtered_tags")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    @log_prompt
    def execute(
        cls, text: str, blacklist_tags: str = "", fixed_tags: str = ""
    ) -> tuple[str, str]:
        """Filter blacklisted tags from a prompt."""
        # 1. Split tokens by BREAK
        text = cls.adjust_fixed_tags(text, fixed_tags)
        groups = text.split("BREAK")
        fixed_tags_set = set(
            [
                cls.normalize_tag(t)
                for t in re.split(r"BREAK|,", fixed_tags)
                if t.strip()
            ]
        )

        # 2. Compile blacklist
        # Convert wildcards to regex
        with open(WILDCARD_PATH, "r") as f:
            wildcards = yaml.safe_load(f)
            for key, values in wildcards.items():
                blacklist_tags = re.sub(
                    f"__{key}__", f"({'|'.join(values)})", blacklist_tags
                )
        compiled_blacklist = re.compile(
            r"|".join([t.strip() for t in blacklist_tags.split(",")])
        )

        # 3. Filter tags from blacklist from each group
        filtered_tag_list = []
        new_groups = []
        visited_tags = set()
        for group in groups:
            # Ignore empty tags
            original_tags = []
            for tag in group.split(","):
                if tag.strip() and tag not in visited_tags:
                    visited_tags.add(tag)
                    original_tags.append(tag)
            comp_tags = [
                (idx, cls.normalize_tag(t)) for idx, t in enumerate(original_tags)
            ]
            valid_idxs = []
            for idx, tag in comp_tags:
                if (tag in fixed_tags_set) or not compiled_blacklist.search(tag):
                    valid_idxs.append(idx)
            new_group = ",".join([original_tags[idx] for idx in sorted(valid_idxs)])
            new_groups.append(new_group.strip())
            filtered_tag_list.extend(
                [
                    original_tags[idx].strip()
                    for idx in range(len(original_tags))
                    if idx not in valid_idxs
                ]
            )

        # 4. Join groups by BREAK
        processed_text = "\nBREAK\n\n".join(new_groups)
        filtered_tags = ", ".join(filtered_tag_list)
        return (processed_text, filtered_tags)

    @classmethod
    def IS_CHANGED(
        cls, text: str, blacklist_tags: str = "", fixed_tags: str = ""
    ) -> tuple:
        return (text, blacklist_tags, fixed_tags)


class FilterSubtags(BasePrompt):
    """Filter subtags from a prompt.

    Examples:
        Input: dog, cat, white dog, black cat
        Output: white dog, black cat

        Input: (cat:0.9), (cat:1.1), black cat, (black cat)
        Output: (cat:0.9), (cat:1.1), black cat, (black cat)
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
        },
        "optional": {
            "fixed_tags": ("STRING", {"default": ""}),
        },
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_text", "filtered_tags")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    @log_prompt
    def execute(cls, text: str, fixed_tags: str = "") -> tuple[str, str]:
        """Filter subtags from a prompt."""
        # 1. Split tokens by BREAK
        text = cls.adjust_fixed_tags(text, fixed_tags)
        groups = text.split("BREAK")
        fixed_tags_set = set(
            [
                cls.normalize_tag(t)
                for t in re.split(r"BREAK|,", fixed_tags)
                if t.strip()
            ]
        )

        # 2. filter all subtags from each group
        filtered_tag_list = []
        new_groups = []
        visited_tags = set()
        for group in groups:
            # Ignore empty tags
            original_tags = []
            for tag in group.split(","):
                if tag.strip() and tag not in visited_tags:
                    visited_tags.add(tag)
                    original_tags.append(tag)
            comp_tags = [
                (idx, cls.normalize_tag(t)) for idx, t in enumerate(original_tags)
            ]
            valid_idxs = set()
            for idx, tag in sorted(
                comp_tags, key=lambda x: (len(x[1]), -x[0]), reverse=True
            ):
                if (tag in fixed_tags_set) or not any(
                    tag in comp_tags[valid_idx][1] for valid_idx in valid_idxs
                ):
                    valid_idxs.add(idx)
            new_group = ",".join([original_tags[idx] for idx in sorted(valid_idxs)])
            new_groups.append(new_group.strip())
            filtered_tag_list.extend(
                [
                    original_tags[idx].strip()
                    for idx in range(len(original_tags))
                    if idx not in valid_idxs
                ]
            )

        # 3. Join groups by BREAK
        processed_text = "\nBREAK\n\n".join(new_groups)
        filtered_tags = ", ".join(filtered_tag_list)
        return (processed_text, filtered_tags)

    @classmethod
    def IS_CHANGED(cls, text: str, fixed_tags: str = "") -> tuple:
        return (text, fixed_tags)


class ReplaceUnderscores(BasePrompt):
    """Replace underscores with spaces in a prompt.

    Examples:
        Input: dog_cat_white_dog_black_cat
        Output: dogcatwhitedogblackcat
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
        }
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    @log_prompt
    def execute(cls, text: str) -> tuple[str]:
        """Replace underscores with spaces in a prompt."""
        processed_text = text.replace("_", " ")
        return (processed_text,)

    @classmethod
    def IS_CHANGED(cls, text: str) -> tuple:
        return (text,)


class FixBreakAfterTIPO(BasePrompt):
    """Fix break after TIPO in a prompt.

    Examples:
        Input: tag, (BREAK:-1), tags
        Output: tag BREAK tags
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
        }
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    @log_prompt
    def execute(cls, text: str) -> tuple[str]:
        """Fix break after TIPO in a prompt."""
        # Remove a weight of BREAK (fix TIPO output prompt)
        processed_text = re.sub(r",?\s*\(BREAK:-1\),?\s*", " BREAK ", text)
        return (processed_text,)

    @classmethod
    def IS_CHANGED(cls, text: str) -> tuple:
        return (text,)


class TokenAnalyzer(BasePrompt):
    """Analyze tokens in a prompt."""

    INPUT_TYPES = lambda: {
        "required": {
            "clip": ("CLIP", {"forceInput": True}),
            "text": ("STRING", {"forceInput": True}),
        }
    }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("g_tokens", "g_token_count", "l_tokens", "l_token_count")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    def execute(cls, clip, text) -> tuple[str, str, str, str]:
        if isinstance(text, list):
            # NOTE: unexpected list type handling
            text = ", ".join(text)
        tokens = clip.tokenize(text)

        results = {}
        tokenizer_ids = ["g", "l"]
        for tokenizer_id in tokenizer_ids:
            tokenizer = getattr(clip.tokenizer, f"clip_{tokenizer_id}")

            # Filter out special tokens (start, end, pad)
            # NOTE: tokens[tokenizer_id].shape: (batch_size, seq_len, embedding_dim)
            # NOTE: seq_len: N*77(75 + start_token + end_token)
            tid_weight_pairs = [
                (tid, weight)
                for tid, weight in tokens[tokenizer_id][0]  # [0]: first sample
                if tid
                not in [tokenizer.start_token, tokenizer.end_token, tokenizer.pad_token]
            ]

            token_strs = []
            for (tid, weight), token_str in tokenizer.untokenize(tid_weight_pairs):
                token_strs.append(
                    f"({token_str}:{weight})" if weight != 1 else token_str
                )
            split_tokens = cls._split_tokens_by_break(token_strs)
            results[tokenizer_id] = {
                "tokens": "\n\n".join([" | ".join(tokens) for tokens in split_tokens]),
                "token_count": ", ".join([str(len(tokens)) for tokens in split_tokens]),
            }

        return (
            results["g"]["tokens"],
            results["g"]["token_count"],
            results["l"]["tokens"],
            results["l"]["token_count"],
        )

    @staticmethod
    def _split_tokens_by_break(tokens: list[str]) -> list[list[str]]:
        """Split tokens by BREAK."""
        # NOTE: break token can be different for each tokenizer
        BREAK_TOKEN = "break</w>"

        concat_tokens = []
        cur_tokens = []
        for token in tokens:
            if token == BREAK_TOKEN:
                concat_tokens.append(cur_tokens)
                cur_tokens = []
            else:
                cur_tokens.append(token)
        else:
            concat_tokens.append(cur_tokens)

        return concat_tokens

    @classmethod
    def IS_CHANGED(cls, clip, text) -> tuple:
        return (clip, text)


if __name__ == "__main__":
    result = ProcessTags.execute(
        "fisheye, (BREAK:-1), cat, dogs, (cat:0.9), (cat:1.1), black cat, (black cat)",
        blacklist_tags="__color__ eyes, hello",
        fixed_tags="cat, dogs, (cat:0.9), (cat:1.1), black cat, (black cat)",
        replace_underscores=True,
        filter_tags=True,
        filter_subtags=True,
    )
    print(result)
