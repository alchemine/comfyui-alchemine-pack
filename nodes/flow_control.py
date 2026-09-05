"""Nodes in AlcheminePack/FlowControl."""

from typing import Any
from .lib.utils import any_typ


#################################################################
# Base class
#################################################################
class BaseFlowControl:
    """Base class for Flow Control nodes."""



#################################################################
# Nodes
#################################################################
class LazyExecution(BaseFlowControl):
    """Pass `value` after the `signal` is passed.
    This is useful when you want to control the execution order of the nodes:
    downstream nodes only run once `signal` resolves, and an upstream
    `ExecutionBlocker` on `signal` propagates through to block them.

    `value` is the first input so that muting (bypassing) this node passes
    `value` straight through — handy for priming a graph that would otherwise
    be blocked by `signal` on a cold start.

    Args:
        value (Any): Value to pass.
        signal (Any): Signal to gate the `value` on.
    """

    INPUT_TYPES = lambda: {
        "required": {
            "value": (any_typ, {}),
            "signal": (any_typ, {}),
        },
    }
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/FlowControl"

    @classmethod
    def execute(cls, value: Any, signal: Any) -> tuple[Any]:
        return (value,)
