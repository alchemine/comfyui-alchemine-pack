from .nodes.prompt import RemoveSubtags


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "RemoveSubtags": RemoveSubtags,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveSubtags": "Remove Subtags",
}
