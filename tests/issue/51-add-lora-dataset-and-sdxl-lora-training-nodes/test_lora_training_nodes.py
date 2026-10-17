import ast
from pathlib import Path

import pytest

PACK_DIR = Path(__file__).resolve().parents[3]
NEW_NODES = ("LoadImagesFromDir", "FilterFrequentTags", "TrainSdxlLora")


def registered_node_ids():
    tree = ast.parse((PACK_DIR / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "NODE_CLASS_MAPPINGS":
            return {key.value for key in node.value.keys}


@pytest.mark.parametrize("node_id", NEW_NODES)
def test_new_node_is_registered(node_id):
    assert node_id in registered_node_ids()


def test_save_image_with_text_keeps_name(pack):
    optional = pack("lora").SaveImageWithText.INPUT_TYPES()["optional"]
    assert {"name", "overwrite"} <= set(optional)


@pytest.mark.parametrize("readme", ("README.md", "README_ko.md"))
def test_readme_documents_new_nodes(readme):
    text = (PACK_DIR / readme).read_text(encoding="utf-8")
    for node_id in NEW_NODES:
        assert f"#### {node_id}" in text
