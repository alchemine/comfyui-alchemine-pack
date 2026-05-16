"""Nodes in AlcheminePack/Input."""

from .lib.utils import exception_handler


#################################################################
# Base class
#################################################################
class BaseInput:
    """Base class for Input nodes."""

    ...


#################################################################
# Nodes
#################################################################
class WidthHeight(BaseInput):
    """Get width and height."""

    INPUT_TYPES = lambda: {
        "required": {
            "width": ("INT", {"default": 512, "min": 1}),
            "height": ("INT", {"default": 512, "min": 1}),
            "swap": ("BOOLEAN", {"default": False}),
            "scale": ("FLOAT", {"default": 1.0, "min": 0.0}),
        }
    }
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Input"

    @classmethod
    def execute(
        cls, width: int = 512, height: int = 512, swap: bool = False, scale: float = 1.0
    ) -> tuple[int, int]:
        width, height = int(width * scale), int(height * scale)
        if swap:
            width, height = height, width
        return width, height


_DEFAULT_EVALUATE_CODE = """def main(tag: str) -> str:
    tags = [t.strip() for t in tag.split(",") if t.strip()]
    return ", ".join(sorted(tags))
"""


class Evaluate(BaseInput):
    """Run user Python code defining main(tag: str) -> str."""

    INPUT_TYPES = lambda: {
        "required": {
            "tag": ("STRING", {"forceInput": True}),
            "code": (
                "STRING",
                {
                    "default": _DEFAULT_EVALUATE_CODE,
                    "multiline": True,
                    "dynamicPrompts": False,
                },
            ),
        },
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tag",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Input"

    @classmethod
    @exception_handler
    def execute(cls, tag: str, code: str) -> tuple[str]:
        ns: dict = {}
        exec(compile(code, "<evaluate_code>", "exec"), ns)
        main = ns.get("main")
        if not callable(main):
            raise TypeError("code must define a callable main(tag: str) -> str")
        out = main(tag)
        if not isinstance(out, str):
            raise TypeError(f"main must return str, got {type(out).__name__}")
        return (out,)

    @classmethod
    def IS_CHANGED(cls, tag: str, code: str) -> tuple:
        return (tag, code)
