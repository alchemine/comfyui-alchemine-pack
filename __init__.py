"""Custom nodes mappings."""

# from .nodes.danbooru import (
#     DanbooruRelatedTagsRetriever,
#     DanbooruPostTagsRetriever,
#     DanbooruPopularPostsTagsRetriever,
#     DanbooruPostsDownloader,
# )
from .nodes.flow_control import SignalSwitch
# from .nodes.inference import GeminiInference, OpenAIInference, OllamaInference, TextEditingInference
from .nodes.input import Evaluate, WidthHeight
from .nodes.io import AsyncSaveImage, PreviewLatestImage
from .nodes.lora import DownloadImage, SaveImageWithText
from .nodes.prompt import (
    ProcessTags,
    FilterTags,
    FilterSubtags,
    ReplaceUnderscores,
    FixBreakAfterTIPO,
    SDXLTokenAnalyzer,
    RemoveWeights,
    SDXLAutoBreak,
    SubstituteTags,
    SeparateLoraTags,
)


NODE_CLASS_MAPPINGS = {
    # # AlcheminePack/Danbooru #########################################################
    # "DanbooruRelatedTagsRetriever": DanbooruRelatedTagsRetriever,
    # "DanbooruPostTagsRetriever": DanbooruPostTagsRetriever,
    # "DanbooruPopularPostsTagsRetriever": DanbooruPopularPostsTagsRetriever,
    # "DanbooruPostsDownloader": DanbooruPostsDownloader,
    # AlcheminePack/FlowControl ######################################################
    "SignalSwitch": SignalSwitch,
    # # AlcheminePack/Inference ########################################################
    # "GeminiInference": GeminiInference,
    # "OpenAIInference": OpenAIInference,
    # "OllamaInference": OllamaInference,
    # "TextEditingInference": TextEditingInference,
    # AlcheminePack/Input ############################################################
    "Evaluate": Evaluate,
    "WidthHeight": WidthHeight,
    # AlcheminePack/IO ###############################################################
    "AsyncSaveImage": AsyncSaveImage,
    "PreviewLatestImage": PreviewLatestImage,
    # AlcheminePack/Lora #############################################################
    "DownloadImage": DownloadImage,
    "SaveImageWithText": SaveImageWithText,
    # AlcheminePack/Prompt #############################################################
    "ProcessTags": ProcessTags,
    "FilterTags": FilterTags,
    "FilterSubtags": FilterSubtags,
    "ReplaceUnderscores": ReplaceUnderscores,
    "FixBreakAfterTIPO": FixBreakAfterTIPO,
    "SDXLTokenAnalyzer": SDXLTokenAnalyzer,
    "RemoveWeights": RemoveWeights,
    "SDXLAutoBreak": SDXLAutoBreak,
    "SubstituteTags": SubstituteTags,
    "SeparateLoraTags": SeparateLoraTags,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # # AlcheminePack/Danbooru #########################################################
    # "DanbooruRelatedTagsRetriever": "Danbooru Related Tags Retriever",
    # "DanbooruPostTagsRetriever": "Danbooru Post Tags Retriever",
    # "DanbooruPopularPostsTagsRetriever": "Danbooru Popular Posts Tags Retriever",
    # "DanbooruPostsDownloader": "Danbooru Posts Downloader",
    # AlcheminePack/FlowControl ######################################################
    "SignalSwitch": "Signal Switch",
    # # AlcheminePack/Inference ########################################################
    # "GeminiInference": "Gemini Inference",
    # "OpenAIInference": "OpenAI Inference",
    # "OllamaInference": "Ollama Inference",
    # "TextEditingInference": "Text Editing Inference",
    # AlcheminePack/Input ############################################################
    "Evaluate": "Evaluate",
    "WidthHeight": "Width Height",
    # AlcheminePack/IO ###############################################################
    "AsyncSaveImage": "Async Save Image",
    "PreviewLatestImage": "Preview Latest Image",
    # AlcheminePack/Lora #############################################################
    "DownloadImage": "Download Image",
    "SaveImageWithText": "Save Image With Text",
    # AlcheminePack/Prompt #############################################################
    "ProcessTags": "Process Tags",
    "FilterTags": "Filter Tags",
    "FilterSubtags": "Filter Subtags",
    "ReplaceUnderscores": "Replace Underscores",
    "FixBreakAfterTIPO": "Fix Break After TIPO",
    "SDXLTokenAnalyzer": "SDXL Token Analyzer",
    "RemoveWeights": "Remove Weights",
    "SDXLAutoBreak": "SDXL Auto Break",
    "SubstituteTags": "Substitute Tags",
    "SeparateLoraTags": "Separate Lora Tags",
}
