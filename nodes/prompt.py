"""Nodes in AlcheminePack/Prompt."""

import random
import re
import numbers
import textwrap
from functools import wraps

import yaml

from .lib.utils import WILDCARD_PATH, get_logger, exception_handler, standardize_prompt
from .lib.tag_guard import (
    filter_generated,
    CATEGORY_NAMES,
)
from .lib.tag_veto import filter_by_veto, veto_available
from .lib.tag_suggest import suggest_tags, suggest_available
from .lib.tag_classify import BUCKETS, classify_tags


logger = get_logger(__file__)

# Danbooru rating names, mildest first; the node also offers "random",
# which draws one of these
RATINGS = ("general", "sensitive", "questionable", "explicit")


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
            if added_texts:
                text = f"{fixed_tags},{added_texts}"
            else:
                text = fixed_tags

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
            n_tokens = count_tokens("g")
            return n_tokens

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


class ConsistencyGuard(BasePrompt):
    """Remove tags contradicting fixed tags (or earlier tags in the text).

    Veto layer over a tag generator: which tag is *best* is subjective,
    which tag is *impossible* is not. A tag is vetoed when its co-occurrence
    lift (observed / expected on 5.48M Danbooru solo posts) with some
    reference tag falls below lift_threshold, with expected co-occurrence
    >= 15 so an observed 0 is evidence, not chance. Composition tags
    (2girls, yuri, ...) are judged on the unfiltered corpus, and
    character-count/gender tags are only ever compared with each other.
    Fixed tags are assumed consistent and always kept; surviving generated
    tags immediately become references, so two mutually contradictory
    suggestions cannot both pass.

    lift_threshold is the filter's only tuned parameter (labeled pairs
    place the contradiction boundary in the 0.098-0.142 gap; raising it
    past ~0.13 starts vetoing compatible pairs).

    Falls back to static category rules (tag_data.py) when the veto
    artifact (tag_veto.npz) is missing.

    Examples:
        Input: text="bikini, waterfall, pond, dress, day, night", fixed_tags="bikini, waterfall, day"
        Output: ("bikini, waterfall, pond, day", <judgment table>)
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
            "lift_threshold": (
                "FLOAT",
                {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01},
            ),
        },
        "optional": {
            "fixed_tags": ("STRING", {"default": ""}),
        },
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_text", "table")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    @log_prompt
    def execute(
        cls,
        text: str,
        lift_threshold: float = 0.1,
        fixed_tags: str = "",
    ) -> tuple[str, str]:
        """Remove tags contradicting fixed tags from a prompt."""
        if veto_available():
            processed_text, filtered_tags = filter_by_veto(
                text,
                fixed_prompt=fixed_tags,
                lift_th=lift_threshold,
            )
        else:
            logger.warning(
                "[ConsistencyGuard] tag_veto.npz not found; "
                "falling back to static category rules"
            )
            processed_text, filtered_tags = filter_generated(
                text,
                locked_prompt=fixed_tags,
                modes={c: "auto" for c in CATEGORY_NAMES},
                clothes_strict=False,
            )
        return (processed_text, filtered_tags)

    @classmethod
    def IS_CHANGED(
        cls,
        text: str,
        lift_threshold: float = 0.1,
        fixed_tags: str = "",
    ) -> tuple:
        return (text, lift_threshold, fixed_tags)


class TagGenerator(BasePrompt):
    """Generate tags that usually accompany the input tags.

    The other direction of ConsistencyGuard's statistic: lift far below 1
    means two tags avoid each other (veto), lift far above 1 means they
    attract. Tags are generated one per step, LM-style: the step
    distribution is naive Bayes over Danbooru solo-post co-occurrence
    (log P(tag) + sum of log lift against every context tag), each pick
    joins the context and re-conditions the next step, and vetoed
    candidates are masked so the output cannot contradict itself.

    temperature 0 = deterministic argmax; above 0, the usual sampling
    filters apply (top_k / top_p / min_p, applied in that order), and
    seed makes the draw reproducible.

    min_count drops rare tags from the candidates, counted within the
    requested rating tier rather than over the whole corpus, so asking
    for a milder rating also shrinks the pool. It defaults to the
    vocabulary floor, i.e. no filtering: three other mechanisms already
    hold rare tags back -- a candidate needs positive attraction from
    the prompt, log P(tag) penalises rare tags heavily, and the stored
    lift is smoothed so a pair seen once cannot look like a strong
    association. Raise it if a particular prompt keeps surfacing tags
    too obscure for your model to have learned.

    categories restricts which knobs the output may turn, using the
    names in resources/group/categories_v1.0.json: characters,
    expressions, pose, clothes, background, compositions, body, objects,
    creatures, etc. "pose, clothes" allows only those two; adding counts
    ("expressions:2, pose:3, background:3") also caps each one, and with
    n 0 those caps become the target length -- the way to get a balanced
    prompt instead of whatever the statistics happen to favour.

    rating caps explicitness on both sides: the statistics come from the
    matching corpus slice, and tags whose own rating level exceeds the
    request are masked, so "general" cannot surface a tag Danbooru only
    applies to racier art. "random" draws one of the four uniformly from
    seed, so the choice is reproducible and a new seed rerolls the
    rating along with the tags.

    n 0 = auto length: a target tag count is drawn from the corpus
    length distribution, and generation also stops early when no
    candidate is at least twice as likely as chance given the context --
    the data has nothing left to say.

    blacklist is a regex matched against each candidate tag (spaced
    form, case-insensitive, substring search): "hair|eyes" drops every
    hair and eye tag, "^black " only the ones starting that way. It
    filters the candidates rather than the result, so n tags still come
    back. An unparseable pattern is logged and ignored.

    rating caps the exposure level of the statistics themselves: the
    co-occurrence tables are built per cumulative rating tier (general <
    sensitive < questionable < explicit, each including the tiers below),
    so at rating "general" the sampler has never seen the associations
    that only exist in racier posts and cannot drift toward them.

    Examples:
        Input: text="night, city, rain", n=5, temperature=0.0
        Output: ("night, city, rain, cityscape, building, night sky, scenery, road",
                 "cityscape, building, night sky, scenery, road")
        Input: text="1girl, beach", n=0, categories="clothes:2, pose:2"
        Output: (..., "swimsuit, bikini, holding swim ring, holding beachball")
        Input: text="1girl, cafe", n=4, blacklist="holding|cup"
        Output: (..., "food, table, chair, plate")
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
            "n": ("INT", {"default": 10, "min": 0, "max": 100}),
            "rating": (
                list(RATINGS) + ["random"],
                {"default": "explicit"},
            ),
            "temperature": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05},
            ),
            "top_k": ("INT", {"default": 50, "min": 0, "max": 500}),
            "top_p": (
                "FLOAT",
                {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
            "min_p": (
                "FLOAT",
                {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
            "seed": (
                "INT",
                {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
            ),
            "min_count": (
                "INT",
                {"default": 100, "min": 100, "max": 1000000, "step": 100},
            ),
        },
        "optional": {
            "categories": ("STRING", {"default": "", "multiline": False}),
            "blacklist": ("STRING", {"default": "", "multiline": False}),
        },
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_text", "generated_tags")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    @log_prompt
    def execute(
        cls,
        text: str,
        n: int = 10,
        rating: str = "explicit",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        min_p: float = 0.0,
        seed: int = 0,
        min_count: int = 100,
        categories: str = "",
        blacklist: str = "",
    ) -> tuple[str, str]:
        """Append companion tags to a prompt."""
        if rating == "random":
            # drawn from seed, so a workflow stays reproducible and a new
            # seed rerolls the rating along with the tags
            rating = random.Random(seed).choice(RATINGS)
            logger.info("[TagGenerator] random rating -> %s", rating)
        rating = rating[0]  # danbooru letter form: g/s/q/e
        if not suggest_available():
            logger.warning(
                "[TagGenerator] suggest artifact not found; passing through"
            )
            return (text, "")
        generated = suggest_tags(
            text, n=n, min_count=min_count,
            temperature=temperature, top_k=top_k, top_p=top_p,
            min_p=min_p, seed=seed, rating=rating, categories=categories,
            blacklist=blacklist,
        )
        generated_str = ", ".join(generated)
        processed = f"{text.strip().rstrip(',')}, {generated_str}" if generated else text
        return (processed, generated_str)

    @classmethod
    def IS_CHANGED(
        cls,
        text: str,
        n: int = 10,
        rating: str = "explicit",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        min_p: float = 0.0,
        seed: int = 0,
        min_count: int = 100,
        categories: str = "",
        blacklist: str = "",
    ) -> tuple:
        return (text, n, rating, temperature, top_k, top_p, min_p,
                seed, min_count, categories, blacklist)


class ClassifyTags(BasePrompt):
    """Split prompt tags into coarse category outputs.

    Buckets come from the same labels TagGenerator samples with
    (resources/group/): the tag's category picks the bucket, and its
    rating level sends questionable and explicit tags to "nsfw"
    instead. Tags the labels do not cover -- about 3% of the
    vocabulary, and well under 1% of real prompt tags by frequency --
    fall back to the static tag_data tables, then to "others".

    Examples:
        Input: text="1boy, serafuku, sitting, smile, classroom"
        Output: characters="1boy", clothes="serafuku", pose="sitting",
                expression="smile", background="classroom", ...
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"forceInput": True}),
        },
    }
    RETURN_TYPES = ("STRING",) * len(BUCKETS)
    RETURN_NAMES = BUCKETS
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    def execute(cls, text: str) -> tuple:
        """Classify tags into buckets and return one string per bucket."""
        buckets = classify_tags(text)
        return tuple(", ".join(buckets[b]) for b in BUCKETS)

    @classmethod
    def IS_CHANGED(cls, text: str) -> tuple:
        return (text,)


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
