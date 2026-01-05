"""Flow Control nodes."""

from typing import Any
from .utils import any_typ


#################################################################
# Base class
#################################################################
class BaseFlowControl:
    """Base class for Flow Control nodes."""

    ...


#################################################################
# Nodes
#################################################################
class SignalSwitch(BaseFlowControl):
    """Pass `value` after the `signal` is passed.
    This is useful when you want to control the execution order of the nodes.

    Args:
        signal (Any): Signal to pass the `value`.
        value (Any): Value to pass.
    """

    INPUT_TYPES = lambda: {
        "required": {
            "signal": (any_typ, {}),
            "value": (any_typ, {}),
        },
    }
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/FlowControl"

    @classmethod
    def execute(cls, signal: Any, value: Any) -> tuple[Any]:
        return (value,)
