import logging
import textwrap
from time import sleep
from pathlib import Path
from functools import wraps


#################################################################
# Logger setup
#################################################################
ROOT_DIR = Path(__file__).parent.parent
CUSTOM_NODES_DIR = ROOT_DIR.parent


def get_logger(name: str = __file__, level: int = logging.WARNING):
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


logger = get_logger()


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
            logger.error(f"# Unexpected error in '{func.__name__}'", exc_info=True)
            raise

    return wrapper


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


def retry(func, retries: int = 3, delay: float = 1e-2, exceptions=(Exception,)):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(retries):
            try:
                result = func(*args, **kwargs)
                break
            except exceptions:
                if attempt < retries - 1:
                    sleep(delay)
                else:
                    raise
        return result

    return wrapper


#################################################################
# Base class
#################################################################
class BasePrompt:
    """Base class for prompt processing.

    This class provides a base class for prompt processing."""

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
