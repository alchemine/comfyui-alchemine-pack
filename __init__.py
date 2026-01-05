"""Custom nodes mappings."""

from .nodes.flow_control import SignalSwitch
from .nodes.prompt import (
    ProcessTags,
    FilterSubtags,
    FilterTags,
    ReplaceUnderscores,
    FixBreakAfterTIPO,
    TokenAnalyzer,
    RemoveWeights,
    AutoBreak,
    SubstituteTags,
)
from .nodes.danbooru import (
    DanbooruRelatedTagsRetriever,
    DanbooruPostTagsRetriever,
    DanbooruPopularPostsTagsRetriever,
    DanbooruPostsDownloader,
)
from .nodes.input import WidthHeight
from .nodes.inference import GeminiInference, OllamaInference, TextEditingInference
from .nodes.lora import DownloadImage, SaveImageWithText
from .nodes.io import AsyncSaveImage, PreviewLatestImage


NODE_CLASS_MAPPINGS = {
    # AlcheminePack/FlowControl #########################################################
    "SignalSwitch": SignalSwitch,
    "ProcessTags": ProcessTags,
    "FilterTags": FilterTags,
    "FilterSubtags": FilterSubtags,
    "ReplaceUnderscores": ReplaceUnderscores,
    "FixBreakAfterTIPO": FixBreakAfterTIPO,
    "TokenAnalyzer": TokenAnalyzer,
    "RemoveWeights": RemoveWeights,
    "AutoBreak": AutoBreak,
    "SubstituteTags": SubstituteTags,
    "DanbooruRelatedTagsRetriever": DanbooruRelatedTagsRetriever,
    "DanbooruPostTagsRetriever": DanbooruPostTagsRetriever,
    "DanbooruPopularPostsTagsRetriever": DanbooruPopularPostsTagsRetriever,
    "DanbooruPostsDownloader": DanbooruPostsDownloader,
    "WidthHeight": WidthHeight,
    "GeminiInference": GeminiInference,
    "OllamaInference": OllamaInference,
    "TextEditingInference": TextEditingInference,
    "DownloadImage": DownloadImage,
    "SaveImageWithText": SaveImageWithText,
    "AsyncSaveImage": AsyncSaveImage,
    "PreviewLatestImage": PreviewLatestImage,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # AlcheminePack/FlowControl #########################################################
    "SignalSwitch": "Signal Switch",
    "ProcessTags": "Process Tags",
    "FilterTags": "Filter Tags",
    "FilterSubtags": "Filter Subtags",
    "ReplaceUnderscores": "Replace Underscores",
    "FixBreakAfterTIPO": "Fix Break After TIPO",
    "TokenAnalyzer": "Token Analyzer",
    "RemoveWeights": "Remove Weights",
    "AutoBreak": "Auto Break",
    "SubstituteTags": "Substitute Tags",
    "DanbooruRelatedTagsRetriever": "Danbooru Related Tags Retriever",
    "DanbooruPostTagsRetriever": "Danbooru Post Tags Retriever",
    "DanbooruPopularPostsTagsRetriever": "Danbooru Popular Posts Tags Retriever",
    "DanbooruPostsDownloader": "Danbooru Posts Downloader",
    "WidthHeight": "Width Height",
    "GeminiInference": "Gemini Inference",
    "OllamaInference": "Ollama Inference",
    "TextEditingInference": "Text Editing Inference",
    "DownloadImage": "Download Image",
    "SaveImageWithText": "Save Image With Text",
    "AsyncSaveImage": "Async Save Image",
    "PreviewLatestImage": "Preview Latest Image",
}
