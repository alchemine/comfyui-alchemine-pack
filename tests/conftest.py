"""The ComfyUI modules the pack imports are stubbed so it imports outside
ComfyUI; nothing else is faked.
"""

import sys
import types
import importlib
from pathlib import Path

import pytest

PACK_DIR = Path(__file__).resolve().parent.parent
PACK_NAME = "alchemine_pack"


@pytest.fixture(scope="session")
def comfy_dirs(tmp_path_factory):
    """ComfyUI's output and user directories, side by side under one root."""
    root = tmp_path_factory.mktemp("comfy")
    dirs = types.SimpleNamespace(root=root, output=root / "output", user=root / "user")
    dirs.output.mkdir()
    (dirs.user / "default" / "workflows").mkdir(parents=True)
    return dirs


@pytest.fixture(scope="session")
def pack(comfy_dirs):
    """Imports a node module by name, the way ComfyUI would."""
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(comfy_dirs.output)
    folder_paths.get_user_directory = lambda: str(comfy_dirs.user)
    sys.modules["folder_paths"] = folder_paths

    graph = types.ModuleType("comfy_execution.graph")
    graph.ExecutionBlocker = type("ExecutionBlocker", (), {})
    sys.modules["comfy_execution"] = types.ModuleType("comfy_execution")
    sys.modules["comfy_execution.graph"] = graph

    package = types.ModuleType(PACK_NAME)
    package.__path__ = [str(PACK_DIR)]
    sys.modules[PACK_NAME] = package
    return lambda name: importlib.import_module(f"{PACK_NAME}.nodes.{name}")
