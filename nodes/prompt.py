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
    BUCKETS,
    classify_tags,
)
from .lib.tag_veto import filter_by_veto, veto_available
from .lib.tag_suggest import (suggest_tags, suggest_available,
                              DEFAULT_COHESION, DEFAULT_REPEAT_DECAY)


logger = get_logger()

# Danbooru rating names, mildest first; the node also offers "random",
# which draws one of these
RATINGS = ("general", "sensitive", "questionable", "explicit")

# TagGenerator's category widgets, in the order they appear on the node:
# the shares that make a prompt read like a picture rather than a list --
# who is in it, what they are doing and feeling, then their body, what it
# wears, and where it is.
#
# These six are what a prompt actually gets steered by; the label file's
# other categories are not worth a knob each, so background carries them
# (see CATEGORY_GROUPS) and creatures and etc are left out of the spec
# entirely.
#
# Category names are hardcoded rather than read from
# resources/group/categories_v1.0.json because INPUT_TYPES runs at import
# and loading the label tables costs more than this list is worth;
# tag_category.parse_categories resolves them against the file at sample
# time, so a rename there only costs the widget its effect, never an
# error.
CATEGORY_DEFAULTS = {
    "characters": 0.1,
    "pose": 0.2,
    "expressions": 0.1,
    "body": 0.3,
    "clothes": 0.2,
    "background": 0.1,
}
# What each widget actually turns. background stands for the scene around
# the subject: the props in it (objects) and how it is framed
# (compositions). The three share one budget rather than getting one
# each, so background at 0.1 is a tenth of the output for the whole
# setting -- objects alone will happily fill a prompt with furniture.
#
# characters is deliberately NOT in that group. The label file files the
# subject itself there -- 1girl, 1boy, solo, 2girls -- not just who else
# is in the scene, and those tags anchor everything downstream: without a
# gender anchor one male pick pulls the whole draw after it. Sharing the
# scene's single slot left them to lose a coin toss against furniture.
CATEGORY_GROUPS = {
    "background": ("background", "objects", "compositions"),
}

# Two categories are deliberately unreachable, and stay out of the spec
# because parse_categories only allows what it is given: creatures pulls
# toward animal-eared characters the prompt did not ask for, and etc is
# the unlabelled remainder, too scattershot to steer with.

# widget value meaning "allowed, no cap"; 0 turns the category off and a
# fraction caps the category's share of the output (0.3 = 30% of n)
CATEGORY_UNCAPPED = -1.0

# Each category gets two widgets: a toggle under its own name and the
# share under name + this. The toggle is the one people reach for, so it
# wins: off means off whatever the share says, which also lets a share
# be dialled in, switched off, and switched back on unchanged.
_SHARE_SUFFIX = "_share"


def _categories_spec(counts):
    """Build a parse_categories spec from the category widgets.

    A widget the caller left out falls back to its default, so a
    workflow saved before these existed still gets the balanced shares
    rather than an unrestricted draw.

    Every widget uncapped still yields a spec rather than "", because
    the excluded categories have to stay excluded. All-zero would read
    as "nothing allowed", which parse_categories cannot express and
    would silently mean "everything"; it is dropped to the defaults with
    a warning instead.
    """
    counts = {name: (counts.get(name + _SHARE_SUFFIX, default)
                     if counts.get(name, True) else 0.0)
              for name, default in CATEGORY_DEFAULTS.items()}
    if all(v == 0.0 for v in counts.values()):
        logger.warning(
            "[TagGenerator] every category is off; sampling with the "
            "defaults instead -- switch at least one back on",
        )
        counts = dict(CATEGORY_DEFAULTS)
    parts = []
    for name, value in counts.items():
        if value == 0.0:
            continue
        # the categories a widget stands for are joined into one budget,
        # so background at 0.1 is a tenth of the output for the whole
        # setting rather than a tenth each for background, objects and
        # compositions
        group = "+".join(CATEGORY_GROUPS.get(name, (name,)))
        parts.append(group if value < 0 else f"{group}:{value}")
    return ", ".join(parts)


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
            blacklist_tags = blacklist_tags.replace(form.format(key=key),
                                                    joined)
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
            
                cls.normalize_tag(t)
                for t in re.split(r"BREAK|,", fixed_tags)
                if t.strip()
            
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
            
                cls.normalize_tag(t)
                for t in re.split(r"BREAK|,", fixed_tags)
                if t.strip()
            
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
                modes=dict.fromkeys(CATEGORY_NAMES, "auto"),
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


def _split_tags(text):
    """Comma-separated prompt into a list of non-empty tags."""
    return [t.strip() for t in text.split(",") if t.strip()]


def _escape_brackets(tag):
    """Escape brackets in a generated tag so they stay literal.

    Every tag here comes from the Danbooru vocabulary, where a bracket
    is part of the name ("star (sky)", "ganyu (genshin impact)") and
    never emphasis, so escaping is unambiguous -- and necessary, since
    attention parsing would otherwise read the qualifier as a weight.
    """
    return re.sub(r"(?<!\\)([()\[\]])", r"\\\1", tag)


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

    Five widgets restrict which knobs the output may turn: pose,
    expressions, body, clothes and background. Each takes a share of the
    output relative to the others, not a fraction of n: with only pose
    0.2 and expressions 0.1 switched on, a request for 10 tags comes
    back 7 pose and 3 expressions, because switching a category off
    hands its share to the ones still on rather than shrinking the
    result. -1 means allowed with no share of its own, 0 switches the
    category off, and the defaults -- body 0.3, pose 0.2, clothes 0.2,
    expressions 0.1, background 0.1 -- balance the node out of the box.

    Counts are split by largest remainder, so they add up to exactly
    what was asked for, and they apply just as well at n 0, where what
    they divide up is the target length drawn from the corpus. They are
    still ceilings rather than promises: a category with nothing left to
    say stops early, and the output comes back short with a warning.

    The widgets do not map one-to-one onto the categories in
    resources/group/categories_v1.0.json. background is the setting
    around the subject, so objects and compositions draw on its budget
    with it -- one share for the scene, not one each. creatures and etc
    are not exposed at all and never sampled.

    Capping matters because each pick re-conditions the next: choosing
    "office chair" makes "swivel chair" more likely, not less, so an
    unrestricted draw tends to pile up in whichever category the prompt
    pulls hardest. background 0.1 breaks that up.

    Three of them are easy to misread. pose owns the sex act groups, but
    explicitness is rating's job, not this one -- leaving pose on at
    rating "general" cannot surface them. clothes owns the job tags, so
    turning it off also drops "office lady" and "nurse". characters owns
    the subject itself, not just the company it keeps: it decides
    whether "1girl" and "solo" can appear at all, which is what anchors
    the gender of everything drawn after them -- and, less happily, the
    franchise grouping tags filed beside them.

    rating caps explicitness on both sides: the statistics come from the
    matching corpus slice, and tags whose own rating level exceeds the
    request are masked, so "general" cannot surface a tag Danbooru only
    applies to racier art. "random" draws one of the four uniformly from
    seed, so the choice is reproducible and a new seed rerolls the
    rating along with the tags.

    lift_threshold is the veto's only tuned parameter, the same knob
    ConsistencyGuard exposes and the same default: a candidate is banned
    when the corpus expected it alongside a reference tag often enough
    (>= 15 posts) and the pair still came in below this fraction of
    chance. 0.1 only catches pairs that essentially never co-occur, so
    raise it when the output keeps contradicting the prompt in ways the
    data merely discourages -- "no panties" and "lace-trimmed panties"
    sit at 0.15, six times rarer than chance but past a 0.1 cut. Vetoed
    candidates are replaced rather than dropped, so n still holds.

    repeat_decay is the brake on cohesion. Because a pick re-conditions
    the next one, the strongest neighbours of "blue skin" are the other
    skin colours, and a draw can spend half its budget enumerating one
    noun -- which the category quotas cannot stop, since those tags all
    sit in the same category. Each tag already in the prompt ending in
    the same word multiplies a candidate's odds by repeat_decay, so the
    second one needs twice the evidence and the third four times.

    cohesion is how much each generated tag conditions the ones after
    it, against the prompt's own pull. At 0 every tag answers to the
    prompt alone and they have nothing to do with each other -- for
    "night, city, rain", mechanical arms beside oversized wings beside a
    leg tattoo. At 1.0 a pick counts for as much as a prompt tag and the
    output reads as one scene, at the risk of becoming its own subject:
    "chair" pulls "office chair" pulls "computer keyboard" until the bar
    the prompt asked for is an office. The default sits between them,
    where a raincoat can follow an umbrella but a category quota still
    stops any one axis from taking the prompt over.

    n 0 = auto length: a target tag count is drawn from the corpus
    length distribution, and generation also stops early when no
    candidate is at least twice as likely as chance given the context --
    the data has nothing left to say.

    blacklist is a regex matched against each candidate tag (spaced
    form, case-insensitive, substring search): "hair|eyes" drops every
    hair and eye tag, "^black " only the ones starting that way. It
    filters the candidates rather than the result, so n tags still come
    back. An unparseable pattern is logged and ignored. The same regex
    is handed to FilterTags below, so one field bans a tag on both ends
    -- write alternatives with "|" rather than commas to keep the two
    reading it the same way.

    rating caps the exposure level of the statistics themselves: the
    co-occurrence tables are built per cumulative rating tier (general <
    sensitive < questionable < explicit, each including the tiers below),
    so at rating "general" the sampler has never seen the associations
    that only exist in racier posts and cannot drift toward them.

    The generated tags then go through the ProcessTags pipeline --
    replace_underscores, filter_tags, filter_subtags, each switchable.
    It runs over the whole prompt with `text` as the fixed tags, so the
    input is never filtered and only the generated tags are at risk. n
    counts what survives: post-processing can drop a pick (a blacklist hit,
    or a tag the prompt already implies, like "dog" once "white dog" is
    there), so the sampler is asked again for as many as went missing.
    That works because it is deterministic given the seed and extends
    its own prefix rather than redrawing, so a top-up round only ever
    adds tags. It gives up after a few rounds and logs how many it got.

    Examples:
        Input: text="night, city, rain", n=5, temperature=0.0
        Output: "night, city, rain, cityscape, building, night sky, scenery, road"
        Input: text="1girl, beach", n=8, clothes=0.25, pose=0.25, rest 0
        Output: "1girl, beach, swimsuit, bikini, holding swim ring, holding beachball"
        Input: text="1girl, cafe", n=4, blacklist="holding|cup"
        Output: "1girl, cafe, food, table, chair, plate"
    """

    # Every widget carries its own one-liner: the docstring above is the
    # reference, but nobody reads it with the node in front of them.
    DESCRIPTION = (
        "Extends a prompt with tags that go with it, drawn from Danbooru "
        "co-occurrence statistics.\n\n"
        "The sampler picks one tag at a time. A tag is a candidate only if "
        "something in the prompt pulls it, it scores by how much rarer than "
        "chance that pull is, and tags the corpus shows the prompt avoiding "
        "are removed outright.\n\n"
        "Start with n and the six category shares; the rest are for when "
        "the output is wrong in a specific way. Hover any widget for what "
        "it does."
    )

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {
                "forceInput": True,
                "tooltip": "The prompt to extend. Its tags condition every "
                           "pick and are never filtered themselves. Tags "
                           "outside the 20,811-tag vocabulary are ignored "
                           "silently -- Danbooru spells a bar "
                           "'bar_(place)', not 'bar'.",
            }),
            "n": ("INT", {
                "default": 20, "min": 0, "max": 100,
                "tooltip": "How many tags to add, counted after "
                           "post-processing. 0 = auto: the length is drawn "
                           "from the corpus and generation also stops early "
                           "once nothing is clearly better than chance.",
            }),
            **{
                key: widget
                for name, share in CATEGORY_DEFAULTS.items()
                for key, widget in (
                    (name, ("BOOLEAN", {
                        "default": True,
                        "tooltip": "Allow %s tags at all. Switching it off "
                                   "hands its share to the categories still "
                                   "on rather than shrinking the output."
                                   % name,
                    })),
                    (name + _SHARE_SUFFIX, ("FLOAT", {
                        "default": share, "min": CATEGORY_UNCAPPED,
                        "max": 1.0, "step": 0.05,
                        "tooltip": "How much of the output %s may take, "
                                   "relative to the other categories that "
                                   "are on: with only pose 0.2 and "
                                   "expressions 0.1, ten tags come back 7 "
                                   "and 3. -1 = allowed with no share of "
                                   "its own." % name,
                    })),
                )
            },
            "lift_threshold": ("FLOAT", {
                "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01,
                "tooltip": "Veto strength. A candidate is banned when the "
                           "corpus expected it alongside a prompt tag often "
                           "enough (>= 15 posts) and it still came in below "
                           "this fraction of chance. Raise it when the "
                           "output contradicts the prompt in ways the data "
                           "merely discourages; 0.1 only catches pairs that "
                           "essentially never co-occur.",
            }),
            "cohesion": ("FLOAT", {
                "default": DEFAULT_COHESION, "min": 0.0, "max": 1.0,
                "step": 0.05,
                "tooltip": "How much each generated tag conditions the ones "
                           "after it. 0 = every tag answers to the prompt "
                           "alone and they have nothing to do with each "
                           "other. 1 = a pick counts as much as a prompt "
                           "tag, so the output reads as one scene but can "
                           "wander off into its own subject.",
            }),
            "repeat_decay": ("FLOAT", {
                "default": DEFAULT_REPEAT_DECAY, "min": 0.05, "max": 1.0,
                "step": 0.05,
                "tooltip": "Shrink a tag's odds by this factor for every "
                           "tag already in the prompt ending in the same "
                           "word. 0.5 halves them each time, so a second "
                           "'<colour> skin' needs twice the evidence the "
                           "first did and a third needs four times; 1.0 "
                           "turns it off. Counters cohesion, which pulls "
                           "hardest along the axis it just moved on.",
            }),
            "rating": (list(RATINGS) + ["random"], {
                "default": "explicit",
                "tooltip": "Explicitness ceiling, on both halves of the "
                           "statistic: the co-occurrence tables come from "
                           "the matching corpus slice, and tags rated above "
                           "the request are masked. It is a ceiling, not a "
                           "target -- 'explicit' permits rather than "
                           "pushes. 'random' draws one from the seed.",
            }),
            "temperature": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                "tooltip": "Sampling randomness. 0 = always take the best "
                           "candidate, which makes the seed irrelevant and "
                           "every run identical. Higher spreads the picks "
                           "over weaker candidates.",
            }),
            "top_k": ("INT", {
                "default": 50, "min": 0, "max": 500,
                "tooltip": "Sample from this many best candidates per step. "
                           "0 = no limit. Ignored at temperature 0.",
            }),
            "top_p": ("FLOAT", {
                "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Keep the best candidates adding up to this much "
                           "probability. 1.0 = no limit. Watch out for 0, "
                           "which leaves exactly one candidate and turns "
                           "sampling back into greedy picking.",
            }),
            "min_p": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Drop candidates below this fraction of the best "
                           "candidate's probability. 0 = off.",
            }),
            "min_count": ("INT", {
                "default": 100, "min": 100, "max": 1000000, "step": 100,
                "tooltip": "Ignore tags with fewer than this many posts in "
                           "the requested rating tier. The default is the "
                           "vocabulary floor, i.e. no filtering. Raise it "
                           "when a prompt keeps surfacing tags too obscure "
                           "for your model to have learned.",
            }),
        },
        "optional": {
            "replace_underscores": ("BOOLEAN", {
                "default": True,
                "tooltip": "Write tags as 'blue eyes' rather than "
                           "'blue_eyes'.",
            }),
            "filter_tags": ("BOOLEAN", {
                "default": True,
                "tooltip": "Drop duplicates and blacklisted tags from the "
                           "finished prompt.",
            }),
            "filter_subtags": ("BOOLEAN", {
                "default": True,
                "tooltip": "Drop tags another tag already implies, keeping "
                           "'white dog' over 'dog'. It can eat a pick the "
                           "sampler just made, which is why the node asks "
                           "for replacements until n survive.",
            }),
            "blacklist": ("STRING", {
                "default": "", "multiline": False,
                "tooltip": "Regex matched against each candidate tag in "
                           "spaced form, case-insensitively: 'hair|eyes' "
                           "drops every hair and eye tag. It filters "
                           "candidates rather than results, so n tags still "
                           "come back. Use '|', not commas.",
            }),
            "seed": (
                "INT",
                {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                 "control_after_generate": True,
                 "tooltip": "Reproducibility. The same seed and settings "
                            "always give the same tags -- unless "
                            "temperature is 0, where the seed does nothing "
                            "at all."},
            ),
        },
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    # each top-up round asks the sampler for more tags; bounded so an
    # over-aggressive filter cannot spin here forever
    _MAX_ROUNDS = 5

    # ceiling on a top-up ask, as a multiple of n: past this the filters
    # are eating so much that drawing more is not the answer
    _MAX_ASKED_FACTOR = 8

    @classmethod
    def _postprocess(cls, prompt, text, blacklist, replace_underscores,
                     filter_tags, filter_subtags):
        """Run the ProcessTags pipeline over a prompt.

        `blacklist` serves both ends of the node -- the sampler masks the
        candidates it matches, FilterTags removes anything that slips
        through -- and the node's own `text` is what FilterTags and
        FilterSubtags treat as fixed, so post-processing only ever
        touches the generated tags.
        """
        if not (replace_underscores or filter_tags or filter_subtags):
            return prompt
        return ProcessTags.execute(
            text=prompt,
            replace_underscores=replace_underscores,
            filter_tags=filter_tags,
            filter_subtags=filter_subtags,
            auto_break=False,
            blacklist_tags=blacklist,
            fixed_tags=text,
        )[0]

    @classmethod
    def _fill(cls, n, draw, process, base, seen):
        """Draw until `n` tags survive post-processing, or rounds run out.

        n is a promise about the finished prompt, not about the draw:
        post-processing runs inside the node precisely so the count asked
        for is the count returned, which means the shortfall it leaves
        has to be redrawn here.

        A bigger ask is not a superset of a smaller one -- the category
        quotas are shares of the ask, so raising it re-splits the budget
        and can even come back with fewer tags -- so each round is judged
        on its own and the best one wins.
        """
        wanted, asked, kept = max(n, 0), max(n, 0), []
        for attempt in range(cls._MAX_ROUNDS if wanted else 1):
            generated = draw(asked if wanted else n)
            processed = process(f"{base.strip().rstrip(',')}, "
                                + ", ".join(_escape_brackets(t)
                                            for t in generated))
            survived = [t for t in _split_tags(processed) if t not in seen]
            if len(survived) > len(kept):
                kept = survived
            if not wanted or len(kept) >= wanted:
                break
            # scale the next ask by the share of the last one that
            # survived rather than by the shortfall: the filters drop a
            # roughly constant fraction, so topping up by the missing
            # count alone gains a round at a time and runs out of rounds
            # before it converges. +attempt keeps it moving when nothing
            # was filtered and the sampler is the one falling short.
            asked = min(-(-asked * wanted // max(len(survived), 1))
                        + attempt + 1,
                        cls._MAX_ASKED_FACTOR * wanted + 16)
        if wanted and len(kept) < wanted:
            logger.warning(
                "[TagGenerator] only %d of %d tags after %d rounds -- lower "
                "lift_threshold or min_count, or relax blacklist and "
                "categories", len(kept), wanted, cls._MAX_ROUNDS,
            )
        return kept[:wanted] if wanted else kept

    @classmethod
    @exception_handler
    @log_prompt
    def execute(
        cls,
        text: str,
        n: int = 10,
        lift_threshold: float = 0.1,
        rating: str = "explicit",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        min_p: float = 0.0,
        seed: int = 0,
        min_count: int = 100,
        blacklist: str = "",
        replace_underscores: bool = True,
        filter_tags: bool = True,
        filter_subtags: bool = True,
        cohesion: float = DEFAULT_COHESION,
        repeat_decay: float = DEFAULT_REPEAT_DECAY,
        **categories: float,
    ) -> tuple[str]:
        """Append companion tags to a prompt."""
        if rating == "random":
            # drawn from seed, so a workflow stays reproducible and a new
            # seed rerolls the rating along with the tags
            rating = random.Random(seed).choice(RATINGS)
            logger.debug("[TagGenerator] random rating -> %s", rating)
        rating = rating[0]  # danbooru letter form: g/s/q/e
        spec = _categories_spec(categories)

        def process(prompt):
            return cls._postprocess(prompt, text, blacklist,
                                    replace_underscores, filter_tags,
                                    filter_subtags)

        if not suggest_available():
            logger.warning(
                "[TagGenerator] suggest artifact not found; passing through"
            )
            return (process(text),)

        # the sampler masks candidates with a single regex, so the widget
        # value has to be compiled the same way FilterTags compiles it --
        # wildcards expanded, commas turned into alternation. Passing it
        # raw made every comma-separated blacklist match nothing, which
        # left the whole list to the post-filter: the sampler kept
        # spending picks on tags that were about to be thrown away.
        blacklist_rx = blacklist_pattern(blacklist)

        # No quota_total: the category quotas are shares of the round's
        # own ask. Pinning them to n would cap the whole draw at n tags,
        # so a top-up round asking for more would get the same list back
        # and the shortfall could never be refilled.
        def draw(m):
            return suggest_tags(
                text, n=m, min_count=min_count, temperature=temperature,
                top_k=top_k, top_p=top_p, min_p=min_p, seed=seed,
                rating=rating, categories=spec, blacklist=blacklist_rx,
                lift_th=lift_threshold,
                cohesion=cohesion, repeat_decay=repeat_decay,
            )

        # the pipeline is applied to the whole prompt, so the kept tags are
        # whatever the combined pass adds on top of the processed input
        base = process(text)
        base_tags = _split_tags(base)
        kept = cls._fill(n, draw, process, base, set(base_tags))
        if not kept:
            return (base,)
        return (", ".join(base_tags + kept),)

    @classmethod
    def IS_CHANGED(
        cls,
        text: str,
        n: int = 10,
        lift_threshold: float = 0.1,
        rating: str = "explicit",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        min_p: float = 0.0,
        seed: int = 0,
        min_count: int = 100,
        blacklist: str = "",
        replace_underscores: bool = True,
        filter_tags: bool = True,
        filter_subtags: bool = True,
        cohesion: float = DEFAULT_COHESION,
        repeat_decay: float = DEFAULT_REPEAT_DECAY,
        **categories: float,
    ) -> tuple:
        return (text, n, lift_threshold, rating, temperature, top_k, top_p,
                min_p, seed, min_count, blacklist, replace_underscores,
                filter_tags, filter_subtags, cohesion, repeat_decay,
                tuple(sorted(categories.items())))

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


class TextPrompt(BasePrompt):
    """Plain text input node without dynamicPrompts, so {a|b} syntax doesn't
    cause the cursor to jump to the end while typing.

    The {option1|option2|...} wildcard expansion is resolved on the Python side
    at execution time (random pick per group, supports nesting).
    """

    INPUT_TYPES = lambda: {
        "required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
        },
        "optional": {
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "forceInput": True}),
        },
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Prompt"

    @classmethod
    @exception_handler
    def execute(cls, text: str, seed: int = 0) -> tuple:
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
    def IS_CHANGED(cls, text: str, seed: int = 0) -> tuple:
        return (text, seed)


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
