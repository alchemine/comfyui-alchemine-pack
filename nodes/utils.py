"""Utility module for the Alchemine Pack."""

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
            logger.error(f"# Unexpected error in '{func.__name__}'", exc_info=True)
            raise

    return wrapper
