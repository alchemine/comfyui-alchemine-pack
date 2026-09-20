"""Nodes in AlcheminePack/Prompt."""

import re
import numbers
import textwrap
from functools import wraps

import yaml

from .lib.utils import WILDCARD_PATH, get_logger, exception_handler, standardize_prompt
from .lib.tag_boy import filter_boy_subject, DEFAULT_ADD_TAGS


logger = get_logger()


#################################################################
# Utility functions
#################################################################
# Wildcard form the blacklist expands: <color> becomes every value under
# the "color" key of resources/wildcards.yaml, joined into one
# alternation. Deliberately NOT the __color__ form the wildcard packs
# use -- their processors run upstream of this node and would resolve the
# token first, and they resolve it by *picking one* value, which is the
# opposite of what a blacklist wants. Angle brackets are syntax no
# wildcard processor claims, so the token survives them and arrives here
# intact. __key__ is still accepted for prompts that never pass through
# one.
_WILDCARD_FORMS = ("<{key}>", "__{key}__")


def blacklist_pattern(blacklist_tags: str) -> str:
    """Comma-separated blacklist -> one regex, or "" when it is empty.

    Each comma-separated token is its own regex ("tan$", "^solo$"), so
    they are joined with | rather than matched as one string. A token
    that will not compile is reported and matched literally instead, so
    one typo cannot silence the whole blacklist.
    """
    if not blacklist_tags or not blacklist_tags.strip():
        return ""
    with open(WILDCARD_PATH) as f:
        wildcards = yaml.safe_load(f) or {}
    for key, values in wildcards.items():
        joined = f"({'|'.join(values)})"
        for form in _WILDCARD_FORMS:
            blacklist_tags = blacklist_tags.replace(form.format(key=key), joined)
    patterns = []
    for t in (t.strip() for t in blacklist_tags.split(",")):
        if not t:
            continue
        try:
            re.compile(t)
        except re.error as exc:
            logger.warning(
                f"Invalid regex in blacklist token {t!r}: {exc}. "
                f"Falling back to literal match."
            )
            t = re.escape(t)
        patterns.append(t)
    return "|".join(patterns)


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
                        row = (
                            f"│ {label:<{col_width1 - 2}} │ {wline.ljust(col_width2)} │"
                        )
                    else:
                        row = f"│ {'':<{col_width1 - 2}} │ {wline.ljust(col_width2)} │"
                    out.append(row)
                    first_row = False
            return "\n".join(out)

        # Prepare inputs
        node_label = args[0].__name__
        input_val = kwargs["text"]
        result = func(*args, **kwargs)
        output_val = result[0]

        # NOTE. 2: space for tags
        top = f"┌{'─' * col_width1}┬{'─' * (2 + col_width2)}┐"
        mid = f"├{'─' * col_width1}┼{'─' * (2 + col_width2)}┤"
        bot = f"└{'─' * col_width1}┴{'─' * (2 + col_width2)}┘"

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
        """
        tag = tag.strip()
        if match := re.search(r"^\(([^()]+):([-0-9. ]+)\)$", tag):
            # Example: (cat:1.20)
            tag, weight = match.groups()
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
    def split_tags(text: str) -> list[str]:
        """Split tags by comma, preserving commas inside parentheses.

        Examples:
            Input: "(masterpiece), (best quality:1.2), (highres, absurdres)"
            Output: ["(masterpiece)", " (best quality:1.2)", " (highres, absurdres)"]
        """
        result = []
        depth = 0
        current = ""
        for char in text:
            if char == "(":
                depth += 1
                current += char
            elif char == ")":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                result.append(current)
                current = ""
            else:
                current += char
        if current:
            result.append(current)
        return result

    @classmethod
    def drop_tags(cls, text: str, is_dropped) -> tuple[str, str]:
        """(prompt without the tags `is_dropped` names, those tags).

        `is_dropped` sees every tag once, weight removed, in prompt order
        -- across BREAK too, since the groups describe one picture. BREAK
        keeps its place and the whitespace around it.
        """
        parts = re.split(r"(\s*BREAK\s*)", text)
        dropped = []
        for i in range(0, len(parts), 2):
            kept = []
            for tag in (t.strip() for t in cls.split_tags(parts[i])):
                if tag:
                    (dropped if is_dropped(cls.remove_weight(tag)) else kept).append(
                        tag
                    )
            parts[i] = ", ".join(kept)
        return ("".join(parts), ", ".join(dropped))

    @classmethod
    def preprocess_tags(cls, text: str, fixed_tags: str) -> tuple[str, str]:
        """Adjust fixed tags to be in the same order as tags in the text."""
        # 1. Adjust BREAK
        text = re.sub(r"(\(?BREAK:?[\d.-]*\)?)", "BREAK", text)
        fixed_tags = re.sub(r"(\(?BREAK:?[-\d.]*\)?)", "BREAK", fixed_tags)

        # 2. Unwrap weights
        text = standardize_prompt(text)
        fixed_tags = standardize_prompt(fixed_tags)

        # 3. Adjust fixed tags
        if fixed_tags:
            fixed_tags_set, fixed_tags_map = [], {}
            for t in re.split(r"BREAK|,", fixed_tags):
                if not t.strip():
                    continue
                normalized_tag = cls.normalize_tag(t)
                if normalized_tag not in fixed_tags_map:
                    fixed_tags_set.append(normalized_tag)
                    fixed_tags_map[normalized_tag] = t

            input_tags_set, input_tags_map = [], {}
            for t in re.split(r"BREAK|,", text):
                if not t.strip():
                    continue
                normalized_tag = cls.normalize_tag(t)
                if normalized_tag not in input_tags_map:
                    input_tags_set.append(normalized_tag)
                    input_tags_map[normalized_tag] = t

            added_texts = ",".join(
                [input_tags_map[t] for t in input_tags_set if t not in fixed_tags_set]
            )
            text = f"{fixed_tags},{added_texts}" if added_texts else fixed_tags

        return text, fixed_tags


#################################################################
# Nodes
#################################################################
class ProcessTags(BasePrompt):
    """Full process of tags from a prompt.

    Order of operations: ReplaceUnderscores -> FilterTags -> FilterSubtags -> AutoBreak
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
            "replace_underscores": ("BOOLEAN", {"default": True}),
            "filter_tags": ("BOOLEAN", {"default": True}),
            "filter_subtags": ("BOOLEAN", {"default": True}),
            "auto_break": ("BOOLEAN", {"default": False}),
        },
        "optional": {
            "clip": ("CLIP",),
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
        auto_break: bool = False,
        clip=None,
        blacklist_tags: str = "",
        fixed_tags: str = "",
    ) -> tuple[str, list[str]]:
        """Process tags from a prompt."""
        # Save original separators BEFORE preprocessing (standardize_prompt changes whitespace)
        original_parts = re.split(r"(\s*BREAK\s*)", text)
        separators = original_parts[1::2]

        text, fixed_tags = cls.preprocess_tags(text, fixed_tags)

        filtered_tags_list = []

        if replace_underscores:
            text = ReplaceUnderscores.execute(text=text)[0]

        if filter_tags:
            text, cur_filtered_tags = FilterTags.execute(
                text=text,
                blacklist_tags=blacklist_tags,
                fixed_tags=fixed_tags,
                preprocess=False,
            )
            if cur_filtered_tags:
                filtered_tags_list.append(cur_filtered_tags)

        if filter_subtags:
            text, cur_filtered_tags = FilterSubtags.execute(
                text=text, fixed_tags=fixed_tags, preprocess=False
            )
            if cur_filtered_tags:
                filtered_tags_list.append(cur_filtered_tags)

        if auto_break and clip is not None:
            text = SDXLAutoBreak.execute(clip=clip, text=text)[0]
            # AutoBreak already formats BREAK correctly, no need to re-join
        else:
            # Re-join with original separators (preserve original whitespace around BREAK)
            groups = text.split("BREAK")
            text = groups[0] if groups else ""
            for i, sep in enumerate(separators):
                if i + 1 < len(groups):
                    text += sep + groups[i + 1]

        return (text, filtered_tags_list)

    @classmethod
    def IS_CHANGED(
        cls,
        text: str,
        replace_underscores: bool = True,
        filter_tags: bool = True,
        filter_subtags: bool = True,
        auto_break: bool = False,
        clip=None,
        blacklist_tags: str = "",
        fixed_tags: str = "",
    ) -> bool:
        return (
            text,
            replace_underscores,
            filter_tags,
            filter_subtags,
            auto_break,
            clip,
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
        cls,
        text: str,
        blacklist_tags: str = "",
        fixed_tags: str = "",
        preprocess: bool = True,
    ) -> tuple[str, str]:
        """Filter blacklisted tags from a prompt."""
        # 1. Split tokens by BREAK (preserve surrounding whitespace)
        # Save original separators BEFORE preprocessing (standardize_prompt changes whitespace)
        original_parts = re.split(r"(\s*BREAK\s*)", text)
        separators = original_parts[1::2]  # Original BREAK with surrounding whitespace

        if preprocess:
            text, fixed_tags = cls.preprocess_tags(text, fixed_tags)

        groups = text.split("BREAK")
        fixed_tags_set = {
            cls.normalize_tag(t) for t in re.split(r"BREAK|,", fixed_tags) if t.strip()
        }

        # 2. Compile blacklist
        pattern = blacklist_pattern(blacklist_tags)
        compiled_blacklist = re.compile(pattern if pattern else r"(?!)")

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
                if (
                    (tag in fixed_tags_set)
                    or not blacklist_tags
                    or (blacklist_tags and not compiled_blacklist.search(tag))
                ):
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

        # 4. Join groups by original BREAK separators (preserve whitespace)
        processed_text = new_groups[0] if new_groups else ""
        for i, sep in enumerate(separators):
            if i + 1 < len(new_groups):
                processed_text += sep + new_groups[i + 1]
        # Remove trailing comma before BREAK
        processed_text = re.sub(r",(\s*BREAK)", r"\1", processed_text)
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
    def execute(
        cls, text: str, fixed_tags: str = "", preprocess: bool = True
    ) -> tuple[str, str]:
        """Filter subtags from a prompt."""
        # 1. Split tokens by BREAK (preserve surrounding whitespace)
        # Save original separators BEFORE preprocessing (standardize_prompt changes whitespace)
        original_parts = re.split(r"(\s*BREAK\s*)", text)
        separators = original_parts[1::2]  # Original BREAK with surrounding whitespace

        if preprocess:
            text, fixed_tags = cls.preprocess_tags(text, fixed_tags)

        groups = text.split("BREAK")
        fixed_tags_set = {
            cls.normalize_tag(t) for t in re.split(r"BREAK|,", fixed_tags) if t.strip()
        }

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

        # 3. Join groups by original BREAK separators (preserve whitespace)
        processed_text = new_groups[0] if new_groups else ""
        for i, sep in enumerate(separators):
            if i + 1 < len(new_groups):
                processed_text += sep + new_groups[i + 1]
        # Remove trailing comma before BREAK
        processed_text = re.sub(r",(\s*BREAK)", r"\1", processed_text)
        filtered_tags = ", ".join(filtered_tag_list)
        return (processed_text, filtered_tags)

    @classmethod
    def IS_CHANGED(cls, text: str, fixed_tags: str = "") -> tuple:
        return (text, fixed_tags)


class ReplaceUnderscores(BasePrompt):
    """Replace underscores with spaces in a prompt.

    Examples:
        Input: dog_cat_white_dog_black_cat
        Output: dog cat white dog black cat
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
    """Fix break after TIPO in a prompt."""

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
        # Step 1: Replace (BREAK:-1) with BREAK
        processed_text = text.replace("(BREAK:-1)", "BREAK")
        # Step 2: Remove commas around BREAK (preserve whitespace)
        processed_text = re.sub(r",(\s*BREAK)", r"\1", processed_text)
        processed_text = re.sub(r"(BREAK\s*),", r"\1", processed_text)
        return (processed_text,)

    @classmethod
    def IS_CHANGED(cls, text: str) -> tuple:
        return (text,)


class SDXLTokenAnalyzer(BasePrompt):
    """Analyze tokens in a prompt (SDXL only - requires clip_l and clip_g)."""

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

        # Split text by BREAK first, then tokenize each part separately
        # This avoids CLIP's internal 77-token chunking which limits each chunk to 75 tokens
        prompts = [p.strip() for p in text.split("BREAK")]

        results = {}
        tokenizer_ids = ["g", "l"]
        for tokenizer_id in tokenizer_ids:
            tokenizer = getattr(clip.tokenizer, f"clip_{tokenizer_id}")

            # Filter out special tokens (start, end, pad)
            # NOTE: tokens[tokenizer_id].shape: (batch_size, seq_len, embedding_dim)
            # NOTE: seq_len: N*77(75 + start_token + end_token)
            # NOTE: tid can be a Tensor for embeddings, so we check if it's an integer first
            special_tokens = [
                tokenizer.start_token,
                tokenizer.end_token,
                tokenizer.pad_token,
            ]

            all_token_strs = []  # List of token lists for each prompt segment

            for prompt in prompts:
                if not prompt:
                    all_token_strs.append([])
                    continue

                tokens = clip.tokenize(prompt)

                # Separate embeddings (Tensors) from regular token IDs
                # NOTE: tid can be a Tensor for embeddings, so we check if it's an integer first
                tid_weight_pairs = []
                embedding_indices = []
                for idx, (tid, weight) in enumerate(tokens[tokenizer_id][0]):
                    if not isinstance(tid, numbers.Integral):
                        # Embedding tensor - mark position
                        embedding_indices.append(len(tid_weight_pairs))
                        tid_weight_pairs.append((tid, weight))
                    elif tid not in special_tokens:
                        tid_weight_pairs.append((tid, weight))

                # Build token strings, handling embeddings separately
                token_strs = []
                embedding_idx_set = set(embedding_indices)
                untokenize_pairs = [
                    (tid, weight)
                    for i, (tid, weight) in enumerate(tid_weight_pairs)
                    if i not in embedding_idx_set
                ]

                untokenize_result = list(tokenizer.untokenize(untokenize_pairs))
                untokenize_iter = iter(untokenize_result)

                for i, (tid, weight) in enumerate(tid_weight_pairs):
                    if i in embedding_idx_set:
                        # Embedding - show placeholder
                        token_str = "[emb]"
                        token_strs.append(
                            f"({token_str}:{weight})" if weight != 1 else token_str
                        )
                    else:
                        (_, _), token_str = next(untokenize_iter)
                        token_strs.append(
                            f"({token_str}:{weight})" if weight != 1 else token_str
                        )

                all_token_strs.append(token_strs)

            results[tokenizer_id] = {
                "tokens": "\n\n".join([" | ".join(t) for t in all_token_strs]),
                "token_count": ", ".join([str(len(t)) for t in all_token_strs]),
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


class RemoveWeights(BasePrompt):
    """Remove weights from a prompt."""

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
    def execute(cls, text: str) -> tuple[str]:
        """Remove weights from a prompt."""
        # Split by BREAK (preserve surrounding whitespace)
        parts = re.split(r"(\s*BREAK\s*)", text)
        groups = parts[::2]  # Even indices: actual groups
        separators = parts[1::2]  # Odd indices: BREAK with surrounding whitespace

        new_groups = []
        for group in groups:
            tags = [cls.remove_weight(t) for t in cls.split_tags(group) if t.strip()]
            new_groups.append(", ".join(tags))

        # Join groups by original BREAK separators (preserve whitespace)
        processed_text = new_groups[0] if new_groups else ""
        for i, sep in enumerate(separators):
            if i + 1 < len(new_groups):
                processed_text += sep + new_groups[i + 1]
        # Remove trailing comma before BREAK
        processed_text = re.sub(r",(\s*BREAK)", r"\1", processed_text)

        return (processed_text,)

    @classmethod
    def IS_CHANGED(cls, text: str) -> tuple:
        return (text,)


class FilterColors(BasePrompt):
    """Keep one colour per thing: of "red dress, blue dress", the first.

    A colour is any value under the "color" key of resources/wildcards.yaml,
    the list FilterTags expands <color> from. What follows the colour names
    the thing, so "black hair, black dress" are two things and both stay.

    Examples:
        Input: 1girl, red dress, blue dress, white shirt, black shirt
        Output: ("1girl, red dress, white shirt", "blue dress, black shirt")
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
        }
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_text", "filtered_tags")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    @log_prompt
    def execute(cls, text: str) -> tuple[str, str]:
        """Keep the first colour of each thing in a prompt."""
        with open(WILDCARD_PATH) as f:
            colors = yaml.safe_load(f)["color"]
        # longest first, so "light blue dress" is not read as a "blue" something
        colors = sorted(colors, key=len, reverse=True)
        pattern = re.compile(rf"({'|'.join(map(re.escape, colors))}) (.+)")
        seen = set()

        def repeated(tag):
            if not (match := pattern.fullmatch(tag)):
                return False
            thing = match.group(2)
            if thing in seen:
                return True
            seen.add(thing)
            return False

        return cls.drop_tags(text, repeated)

    @classmethod
    def IS_CHANGED(cls, text: str) -> tuple:
        return (text,)


class FilterPlurals(BasePrompt):
    """Of two tags that differ only by a plural "s", keep the first.

    "arm up, arms up", "hand on hip, hands on hips": one spelling is
    enough, and the one written first is the one meant. The singular is
    crude -- a trailing "s" comes off any word of four letters or more --
    which is all a comparison between two tags of the same prompt needs:
    "glasses" does not turn into "glass", and "ass" is left alone.

    Examples:
        Input: 1girl, arm up, arms up, smile
        Output: ("1girl, arm up, smile", "arms up")
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
        }
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_text", "filtered_tags")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    @log_prompt
    def execute(cls, text: str) -> tuple[str, str]:
        """Keep the first of two tags that differ only by a plural "s"."""
        first = {}

        def respelled(tag):
            singular = " ".join(
                w[:-1] if len(w) > 3 and w.endswith("s") else w for w in tag.split(" ")
            )
            # the same tag twice is a duplicate, not a plural
            return first.setdefault(singular, tag) != tag

        return cls.drop_tags(text, respelled)

    @classmethod
    def IS_CHANGED(cls, text: str) -> tuple:
        return (text,)


class BoySubjectFilter(BasePrompt):
    """Make the subject tags agree with a tag that needs a man.

    "sex", "hetero", "penis", anything spelled with "another": with one
    of these in the prompt, "solo" is no longer true and is taken out,
    and a prompt that counts no boy gets "((1boy))" and add_tags. A
    prompt without such a tag is left alone.

    Examples:
        Input: text="1girl, solo, sex, smile", add_tags="hetero"
        Output: ("1girl, sex, smile, ((1boy)), hetero",)

        Input: text="1girl, 1boy, solo, sex"
        Output: ("1girl, 1boy, sex",)
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
            "add_tags": ("STRING", {"default": DEFAULT_ADD_TAGS}),
        },
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    @log_prompt
    def execute(cls, text: str, add_tags: str = DEFAULT_ADD_TAGS) -> tuple[str]:
        """Make the subject tags agree with a tag that needs a man."""
        split = lambda s: [t.strip() for t in cls.split_tags(s) if t.strip()]
        return (", ".join(filter_boy_subject(split(text), split(add_tags))),)

    @classmethod
    def IS_CHANGED(cls, text: str, add_tags: str = DEFAULT_ADD_TAGS) -> tuple:
        return (text, add_tags)


class SDXLAutoBreak(BasePrompt):
    """Automatically insert BREAK to keep each segment within 75 tokens (SDXL only - requires clip_g)."""

    INPUT_TYPES = lambda: {
        "required": {
            "clip": ("CLIP", {"forceInput": True}),
            "text": ("STRING", {"forceInput": True}),
        }
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    def execute(cls, clip, text) -> tuple[str]:
        if isinstance(text, list):
            text = ", ".join(text)

        def count(t):
            if not t.strip():
                return 0
            toks = clip.tokenize(t)

            def count_tokens(k):
                tokenizer = getattr(clip.tokenizer, f"clip_{k}")
                special_tokens = [
                    tokenizer.start_token,
                    tokenizer.end_token,
                    tokenizer.pad_token,
                ]
                return sum(
                    1
                    for tid, _ in toks[k][0]
                    if not isinstance(tid, numbers.Integral)
                    or tid not in special_tokens
                )

            # NOTE: use g tokenizer only
            # return max(count_tokens(k) for k in ["g", "l"])
            return count_tokens("g")

        def split(seg):
            # 각 단어와 그 끝 위치 추적 (원본 보존을 위해)
            words = []
            word_ends = []
            for match in re.finditer(r"[^,]+", seg):
                word = match.group().strip()
                if word:
                    words.append(word)
                    word_ends.append(match.end())

            n_words = len(words)
            if n_words >= 2:
                # NOTE: token count 75 can be overflow or fit. But, 'fit' case is ignored.
                if count(seg[: word_ends[n_words - 1]]) >= 75:
                    for i in range(n_words - 1, 0, -1):
                        prefix = seg[: word_ends[i - 1]]  # i번째 단어까지 원본 그대로
                        if count(prefix) < 75:
                            suffix = re.sub(r"^[,\s]*", "", seg[word_ends[i - 1] :])
                            result = f"{prefix}\n\nBREAK\n{split(suffix)}"
                            break
                else:
                    result = seg
            else:
                result = seg
            return result

        # Remove only commas around BREAK (preserve whitespace/newlines)
        result = "BREAK".join(split(s) for s in text.split("BREAK") if s)
        result = re.sub(r",*(\s*)BREAK", r"\1BREAK", result)
        result = re.sub(r"BREAK(\s*),*", r"BREAK\1", result)
        return (result,)

    @classmethod
    def IS_CHANGED(cls, clip, text) -> tuple:
        return (clip, text)


class SubstituteTags(BasePrompt):
    """Replace text using regex pattern with conditional execution.

    Args:
        text: Input text to process
        pattern: Regex pattern to match
        repl: Replacement string
        run_if: Optional regex pattern. Replacement is performed only if this pattern EXISTS.
        skip_if: Optional regex pattern. Replacement is SKIPPED if this pattern exists.

    Examples:
        - "girl이 없으면 1boy → 1girl, 1boy":
          pattern="1boy", repl="1girl, 1boy", skip_if="girl"
        - "1boy가 있으면 solo 제거":
          pattern="solo,?\\s*", repl="", run_if="1boy"
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
            "pattern": (
                "STRING",
                {"default": "", "multiline": False, "placeholder": "regex pattern"},
            ),
            "repl": (
                "STRING",
                {"default": "", "multiline": False, "placeholder": "replacement"},
            ),
        },
        "optional": {
            "run_if": (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "placeholder": "run only if this pattern exists",
                },
            ),
            "skip_if": (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "placeholder": "skip if this pattern exists",
                },
            ),
        },
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    def execute(
        cls, text: str, pattern: str, repl: str, run_if: str = "", skip_if: str = ""
    ) -> tuple[str]:
        """Replace text using regex pattern with conditional execution."""
        # If run_if is provided, only run if the pattern is found
        if run_if and not re.search(run_if, text):
            return (text,)

        # If skip_if is provided, skip replacement if the pattern is found
        if skip_if and re.search(skip_if, text):
            return (text,)

        # Perform the replacement
        processed_text = re.sub(pattern, repl, text)
        return (processed_text,)

    @classmethod
    def IS_CHANGED(
        cls, text: str, pattern: str, repl: str, run_if: str = "", skip_if: str = ""
    ) -> tuple:
        return (text, pattern, repl, run_if, skip_if)


class SeparateLoraTags(BasePrompt):
    """Separate lora tags from a prompt.

    - text_without_lora: input text with all lora tags removed (whitespace preserved as much as possible)
    - text_with_lora: deduplicated lora tags joined by space; if the same lora appears
      multiple times, the last weight wins; original order is preserved

    Examples:
        Input:
            "1girl, <lora:a.safetensors:0.7> blonde, jewelry,
            <lora:b.safetensors:0.7> <lora:c.safetensors:0.7> <lora:c.safetensors:1.0>"
        Output:
            text_without_lora: "1girl, blonde, jewelry"
            text_with_lora: "<lora:a.safetensors:0.7> <lora:b.safetensors:0.7> <lora:c.safetensors:1.0>"
    """

    LORA_PATTERN = re.compile(r"<lora:([^>]+)>")

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
        }
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text_without_lora", "text_with_lora")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    def execute(cls, text: str) -> tuple[str, str]:
        """Separate lora tags from a prompt."""
        # 1. Build text_with_lora: dedupe by lora name, keep last weight, preserve first-seen order
        ordered_names: list[str] = []
        weights: dict[str, str] = {}
        for inner in cls.LORA_PATTERN.findall(text):
            name, _, weight = inner.rpartition(":")
            if not name:
                name, weight = inner, ""
            if name not in weights:
                ordered_names.append(name)
            weights[name] = weight
        text_with_lora = " ".join(
            f"<lora:{name}:{weights[name]}>" if weights[name] else f"<lora:{name}>"
            for name in ordered_names
        )

        # 2. Build text_without_lora using a conditional block rule:
        #    - If a lora block is followed by ',', that trailing comma serves as the separator,
        #      so the preceding "[,\s]*" is consumed along with the lora block.
        #    - Otherwise, only the preceding whitespace is consumed so the preceding comma
        #      can serve as the separator. Trailing whitespace after the block is preserved
        #      in both cases to keep the original spacing intact.
        text_without_lora = re.sub(
            r"[,\s]*<lora:[^>]+>(?:\s+<lora:[^>]+>)*(?=,)", "", text
        )
        text_without_lora = re.sub(
            r"\s*<lora:[^>]+>(?:\s+<lora:[^>]+>)*", "", text_without_lora
        )
        text_without_lora = text_without_lora.strip()
        text_without_lora = re.sub(r"^,\s*", "", text_without_lora)
        text_without_lora = re.sub(r",\s*$", "", text_without_lora)

        return (text_without_lora, text_with_lora)

    @classmethod
    def IS_CHANGED(cls, text: str) -> tuple:
        return (text,)


def _impact_process(text: str, seed: int):
    """Expand text with Impact Pack's wildcard engine, or None if unavailable.

    Impact's process() resolves both {a|b|c} groups and __wildcard__ file
    references (plus $$ multi-select, nesting and # comments) in one pass, so
    a TextPrompt can stand in for an ImpactWildcardProcessor node.
    """
    try:
        from impact.wildcards import process as _process
    except Exception:
        try:
            from ComfyUI_Impact_Pack.modules.impact.wildcards import process as _process
        except Exception:
            return None
    try:
        return _process(text, seed)
    except Exception as e:
        logger.warning(f"Impact wildcard expansion failed, falling back: {e}")
        return None


class TextPrompt(BasePrompt):
    """Plain text input node without dynamicPrompts, so {a|b} syntax doesn't
    cause the cursor to jump to the end while typing.

    Expansion happens on the Python side at execution time. When Impact Pack is
    installed both {option1|option2|...} groups and __wildcard__ file
    references are resolved; otherwise only {a|b|c} groups are, via the
    built-in fallback.

    Leaving `seed` unconnected pins it to 0, so the same text always expands the
    same way; connect a seed to reroll the picks per run.
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
        },
        "optional": {
            "seed": (
                "INT",
                {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "forceInput": True},
            ),
        },
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    def execute(cls, text: str, seed: int | None = None) -> tuple:
        # No seed wired in -> 0, a fixed draw rather than whatever the shared
        # RNG happens to hold. Impact's process() seeds the global random module,
        # so an unseeded call would also drift with unrelated nodes' expansions.
        seed = 0 if seed is None else seed
        expanded = _impact_process(text, seed)
        if expanded is not None:
            return (expanded,)

        import random as _random

        rng = _random.Random(seed)

        def _resolve(s: str) -> str:
            # Iteratively expand innermost {a|b|c} groups until none remain.
            pattern = re.compile(r"\{([^{}]*)\}")
            while True:
                m = pattern.search(s)
                if not m:
                    break
                options = m.group(1).split("|")
                s = s[: m.start()] + rng.choice(options) + s[m.end() :]
            return s

        return (_resolve(text),)

    @classmethod
    def IS_CHANGED(cls, text: str, seed: int | None = None) -> tuple:
        return (text, 0 if seed is None else seed)


if __name__ == "__main__":
    text = "(drunk, beer), full-face blush"
    text = "(happy, drunk, :3), (drunk, beer), full-face blush"
    text = "(happy, drunk, :3:1.3), (beer, can), full-face blush"
    text = "(happy, :3, drunk:1.3), (:>, can, :<), full-face blush"
    text = "(wariza), :3, palace, marble \\(stone\\), curtains, garden, fountain, plant, flower, lanterns"
    text = "blush, \n(covering body, do something),\n\n(:3)"
    result = ProcessTags.execute(
        text,
        fixed_tags=text,
        replace_underscores=True,
        filter_tags=True,
        filter_subtags=True,
    )
    logger.info(result[0])
    logger.info(result[1])
