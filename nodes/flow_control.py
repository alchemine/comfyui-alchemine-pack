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
    """Pass value when the signal is triggered."""

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

    @classmethod
    def IS_CHANGED(cls, signal: Any, value: Any) -> tuple:
        return (signal, value)
