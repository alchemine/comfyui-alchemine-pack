"""Nodes in AlcheminePack/Evaluate."""

from .lib.utils import exception_handler


#################################################################
# Base class
#################################################################
class BaseEvaluate:
    """Base class for Evaluate nodes."""

    ...


#################################################################
# Nodes
#################################################################
_DEFAULT_EVALUATE_CODE = """def main(tag: str) -> str:
    tags = [t.strip() for t in tag.split(",") if t.strip()]
    return ", ".join(sorted(tags))
"""


class Evaluate(BaseEvaluate):
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
    CATEGORY = "AlcheminePack/Evaluate"

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
