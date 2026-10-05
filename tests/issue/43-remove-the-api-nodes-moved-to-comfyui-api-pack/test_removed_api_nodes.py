import ast
from pathlib import Path

import pytest

PACK_DIR = Path(__file__).resolve().parents[3]
MOVED_NODES = (
    "LoadWorkflow",
    "ApiGenerate",
    "ApiSubmit",
    "ApiCollect",
    "GrokGenerate",
    "GrokSubmit",
    "GrokCollect",
    "OpenAIInference",
)
OTHER_PACKS = (
    "comfyui-generator-pack",
    "comfyui-daam-pack",
    "comfyui-danbooru-pack",
    "comfyui-evaluate-pack",
    "comfyui-api-pack",
)


def registered_node_ids():
    tree = ast.parse((PACK_DIR / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "NODE_CLASS_MAPPINGS":
            return {key.value for key in node.value.keys}


def test_moved_node_ids_are_not_registered():
    assert registered_node_ids() & set(MOVED_NODES) == set()


def test_moved_node_files_are_gone():
    paths = (
        "nodes/api.py",
        "nodes/grok.py",
        "nodes/inference.py",
        "nodes/lib/joblock.py",
        ".env.example",
    )

    assert [p for p in paths if (PACK_DIR / p).exists()] == []


def test_dotenv_is_not_required():
    assert "python-dotenv" not in (PACK_DIR / "requirements.txt").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("readme", ["README.md", "README_ko.md"])
def test_readme_links_every_other_pack(readme):
    text = (PACK_DIR / readme).read_text(encoding="utf-8")

    assert [
        p for p in OTHER_PACKS if f"https://github.com/alchemine/{p}" not in text
    ] == []
