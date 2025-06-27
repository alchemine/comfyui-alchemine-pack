from .nodes.prompt import ProcessTags, RemoveSubtags, RemoveTags, RemoveUnderscores


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ProcessTags": ProcessTags,
    "RemoveTags": RemoveTags,
    "RemoveSubtags": RemoveSubtags,
    "RemoveUnderscores": RemoveUnderscores,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProcessTags": "Process Tags",
    "RemoveTags": "Remove Tags",
    "RemoveSubtags": "Remove Subtags",
    "RemoveUnderscores": "Remove Underscores",
}
