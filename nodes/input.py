"""Nodes in AlcheminePack/Input."""


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
