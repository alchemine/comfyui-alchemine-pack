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
            "aspect_ratio": (
                [
                    "832 x 480",
                    "640 x 480",
                    "480 x 640",
                    "480 x 832",
                    "512 x 512",
                    "640 x 640",
                ],
                {"default": "480 x 640"},
            ),
            "scale": ("FLOAT", {"default": 1.0, "min": 1.0}),
            "validate_by_16": ("BOOLEAN", {"default": True}),
        }
    }
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Input"

    @classmethod
    def execute(
        cls,
        aspect_ratio: str = "480 x 640",
        scale: float = 1.0,
        validate_by_16: bool = True,
    ) -> tuple[int, int]:
        width, height = map(int, aspect_ratio.split("x"))
        return int(width * scale), int(height * scale)

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        aspect_ratio: str = "480 x 640",
        scale: float = 1.0,
        validate_by_16: bool = True,
    ) -> bool:
        # Check if width and height are divisible by 16
        width, height = map(int, aspect_ratio.split("x"))
        width, height = int(width * scale), int(height * scale)
        if validate_by_16:
            return width % 16 == 0 and height % 16 == 0
        else:
            return True


if __name__ == "__main__":
    result = WidthHeight.execute(aspect_ratio="480 x 640")
    print(result)
