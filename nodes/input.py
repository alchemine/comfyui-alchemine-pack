from .utils import get_logger

logger = get_logger()


#################################################################
# Base class
#################################################################
class BasePrompt:
    """Base class for Prompt nodes."""

    pass


#################################################################
# Nodes
#################################################################
class WidthHeight(BasePrompt):
    """Get predefined width and height."""

    INPUT_TYPES = lambda: {
        "required": {
            "width": ("INT", {"default": 480, "min": 1}),
            "height": ("INT", {"default": 640, "min": 1}),
            "swap": ("BOOLEAN", {"default": False}),
            "scale": ("FLOAT", {"default": 1.0, "min": 1.0}),
        }
    }
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Input"

    @classmethod
    def execute(
        cls, width: int = 480, height: int = 640, swap: bool = False, scale: float = 1.0
    ) -> tuple[int, int]:
        width, height = int(width * scale), int(height * scale)
        if swap:
            width, height = height, width
        return width, height


if __name__ == "__main__":
    result = WidthHeight.execute(aspect_ratio="480 x 640")
    print(result)
