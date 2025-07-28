from .nodes.prompt import (
    ProcessTags,
    FilterSubtags,
    FilterTags,
    ReplaceUnderscores,
    CustomProcessor,
    TokenAnalyzer,
)
from .nodes.danbooru import (
    DanbooruRelatedTagsRetriever,
    DanbooruPostTagsRetriever,
    DanbooruPopularPostsTagsRetriever,
)
from .nodes.input import WidthHeight
from .nodes.inference import GeminiInference, OllamaInference


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ProcessTags": ProcessTags,
    "FilterTags": FilterTags,
    "FilterSubtags": FilterSubtags,
    "ReplaceUnderscores": ReplaceUnderscores,
    "CustomProcessor": CustomProcessor,
    "TokenAnalyzer": TokenAnalyzer,
    "DanbooruRelatedTagsRetriever": DanbooruRelatedTagsRetriever,
    "DanbooruPostTagsRetriever": DanbooruPostTagsRetriever,
    "DanbooruPopularPostsTagsRetriever": DanbooruPopularPostsTagsRetriever,
    "WidthHeight": WidthHeight,
    "GeminiInference": GeminiInference,
    "OllamaInference": OllamaInference,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProcessTags": "Process Tags",
    "FilterTags": "Filter Tags",
    "FilterSubtags": "Filter Subtags",
    "ReplaceUnderscores": "Replace Underscores",
    "CustomProcessor": "Custom Processor",
    "TokenAnalyzer": "Token Analyzer",
    "DanbooruRelatedTagsRetriever": "Danbooru Related Tags Retriever",
    "DanbooruPostTagsRetriever": "Danbooru Post Tags Retriever",
    "DanbooruPopularPostsTagsRetriever": "Danbooru Popular Posts Tags Retriever",
    "WidthHeight": "Width Height",
    "GeminiInference": "Gemini Inference",
    "OllamaInference": "Ollama Inference",
}
