"""Utility module for the Alchemine Pack."""

import re
import json
import logging
from pathlib import Path
from functools import wraps


#################################################################
# Logger setup
#################################################################
ROOT_DIR = Path(__file__).parent.parent
CUSTOM_NODES_DIR = ROOT_DIR.parent
CONFIG = json.load(open(ROOT_DIR / "config.json"))
RESOURCES_DIR = ROOT_DIR / "resources"
WILDCARD_PATH = RESOURCES_DIR / "wildcards.yaml"


def get_logger(name: str = __file__, level: int = logging.WARNING) -> logging.Logger:
    """Get a logger with a custom formatter that shows the relative path of the file."""

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
            logging.error(f"# Unexpected error in '{func.__name__}'", exc_info=True)
            raise

    return wrapper


attn_syntax = (
    r"\\\(|"
    r"\\\)|"
    r"\\\[|"
    r"\\]|"
    r"\\\\|"
    r"\\|"
    r"\(|"
    r"\[|"
    r":\s*([+-]?[.\d]+)\s*\)|"
    r"\)|"
    r"]|"
    r"[^\\()\[\]:]+|"
    r":"
)
re_attention = re.compile(attn_syntax, re.X)
re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


# NOTE: https://github.com/KohakuBlueleaf/z-tipo-extension/blob/6c6bd9f40bca42f9bbab8b1e7a2ba51cb0d5424b/nodes/tipo.py#L63
def parse_prompt_attention(text):
    r"""
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)
        if text.startswith(r"\\"):
            res.append([text[1:], 1.0])
        elif text == r"(":
            round_brackets.append(len(res))
        elif text == r"[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == r")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == r"]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", 1.0])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def standardize_prompt(text: str) -> str:
    # Handle :3
    text = re.sub(r":3\)", r":3:1.1)", text)
    attentions = parse_prompt_attention(text)
    tags = []
    for tag, weight in attentions:
        tag = tag.strip(", ")
        if not tag:
            continue
        if weight == 1:
            tags.append(tag)
        else:
            for sub_tag in re.split(r",\s*", tag):
                tags.append(f"({sub_tag}:{round(weight, 2)})")
    return ", ".join(tags)


# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")
