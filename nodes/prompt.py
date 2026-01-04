import numbers
import re
import textwrap
from functools import wraps

import yaml

from .utils import WILDCARD_PATH, get_logger, exception_handler, standardize_prompt


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
            text = AutoBreak.execute(clip=clip, text=text)[0]

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
        # 1. Split tokens by BREAK
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

        # 4. Join groups by BREAK
        processed_text = "\nBREAK\n\n".join(new_groups)
        # Remove trailing comma before BREAK
        processed_text = re.sub(r",\s*\n?BREAK", "\nBREAK", processed_text)
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
        # 1. Split tokens by BREAK
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

        # 3. Join groups by BREAK
        processed_text = "\nBREAK\n\n".join(new_groups)
        # Remove trailing comma before BREAK
        processed_text = re.sub(r",\s*\n?BREAK", "\nBREAK", processed_text)
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
        text_groups = []
        groups = text.split("BREAK")
        for group in groups:
            tags = [cls.remove_weight(t) for t in group.split(",") if t.strip()]
            text_groups.append(", ".join(tags))
        processed_text = "\nBREAK\n\n".join(text_groups)
        # Remove trailing comma before BREAK
        processed_text = re.sub(r",\s*\n?BREAK", "\nBREAK", processed_text)

        return (processed_text,)

    @classmethod
    def IS_CHANGED(cls, text: str) -> tuple:
        return (text,)


class AutoBreak(BasePrompt):
    """Automatically insert BREAK to keep each segment within 75 tokens."""

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
    print(result[0])
    print(result[1])
