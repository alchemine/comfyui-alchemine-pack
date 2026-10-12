import ast
from pathlib import Path

import pytest

PACK_DIR = Path(__file__).resolve().parents[3]
OTHER_PACKS = (
    "comfyui-generator-pack",
    "comfyui-daam-pack",
    "comfyui-danbooru-pack",
    "comfyui-evaluate-pack",
    "comfyui-api-pack",
    "comfyui-filter-pack",
)


def registered_node_ids():
    tree = ast.parse((PACK_DIR / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "NODE_CLASS_MAPPINGS":
            return {key.value for key in node.value.keys}


def test_adjust_image_is_not_registered():
    assert "AdjustImage" not in registered_node_ids()


def test_image_module_is_gone():
    assert not (PACK_DIR / "nodes" / "image.py").exists()


@pytest.mark.parametrize("readme", ["README.md", "README_ko.md"])
def test_readme_links_every_other_pack(readme):
    text = (PACK_DIR / readme).read_text(encoding="utf-8")

    assert [
        p for p in OTHER_PACKS if f"https://github.com/alchemine/{p}" not in text
    ] == []


@pytest.mark.parametrize("readme", ["README.md", "README_ko.md"])
def test_readme_does_not_describe_adjust_image(readme):
    text = (PACK_DIR / readme).read_text(encoding="utf-8")

    assert "AdjustImage" not in text
