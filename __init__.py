"""Custom nodes mappings."""

from .nodes.danbooru_requests import (
    DanbooruRelatedTagsRetriever,
    DanbooruPostTagsRetriever,
    DanbooruPopularPostsTagsRetriever,
    DanbooruPostsDownloader,
)
from .nodes.flow_control import LazyExecution
from .nodes.grok import GrokGenerate, GrokSubmit, GrokCollect
from .nodes.inference import OpenAIInference
from .nodes.evaluate import Evaluate
from .nodes.lora import DownloadImage, SaveImageWithText
from .nodes.model import CachedLoraTagLoader
from .nodes.api import LoadWorkflow, ApiGenerate, ApiSubmit, ApiCollect
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
    ConsistencyGuard,
    ClassifyTags,
)


NODE_CLASS_MAPPINGS = {
    # AlcheminePack/Danbooru #########################################################
    "DanbooruRelatedTagsRetriever": DanbooruRelatedTagsRetriever,
    "DanbooruPostTagsRetriever": DanbooruPostTagsRetriever,
    "DanbooruPopularPostsTagsRetriever": DanbooruPopularPostsTagsRetriever,
    "DanbooruPostsDownloader": DanbooruPostsDownloader,
    # AlcheminePack/FlowControl ######################################################
    "LazyExecution": LazyExecution,
    # AlcheminePack/Grok #############################################################
    "GrokGenerate": GrokGenerate,
    "GrokSubmit": GrokSubmit,
    "GrokCollect": GrokCollect,
    # AlcheminePack/Inference ########################################################
    "OpenAIInference": OpenAIInference,
    # AlcheminePack/Evaluate #########################################################
    "Evaluate": Evaluate,
    # AlcheminePack/Lora #############################################################
    "DownloadImage": DownloadImage,
    "SaveImageWithText": SaveImageWithText,
    # AlcheminePack/Model ############################################################
    "CachedLoraTagLoader": CachedLoraTagLoader,
    # AlcheminePack/API ##############################################################
    "LoadWorkflow": LoadWorkflow,
    "ApiGenerate": ApiGenerate,
    "ApiSubmit": ApiSubmit,
    "ApiCollect": ApiCollect,
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
    "ConsistencyGuard": ConsistencyGuard,
    "ClassifyTags": ClassifyTags,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # AlcheminePack/Danbooru #########################################################
    "DanbooruRelatedTagsRetriever": "Danbooru Related Tags Retriever",
    "DanbooruPostTagsRetriever": "Danbooru Post Tags Retriever",
    "DanbooruPopularPostsTagsRetriever": "Danbooru Popular Posts Tags Retriever",
    "DanbooruPostsDownloader": "Danbooru Posts Downloader",
    # AlcheminePack/FlowControl ######################################################
    "LazyExecution": "Lazy Execution",
    # AlcheminePack/Grok #############################################################
    "GrokGenerate": "Grok Generate",
    "GrokSubmit": "Grok Submit",
    "GrokCollect": "Grok Collect",
    # AlcheminePack/Inference ########################################################
    "OpenAIInference": "OpenAI Inference",
    # AlcheminePack/Evaluate #########################################################
    "Evaluate": "Evaluate",
    # AlcheminePack/Lora #############################################################
    "DownloadImage": "Download Image",
    "SaveImageWithText": "Save Image With Text",
    # AlcheminePack/Model ############################################################
    "CachedLoraTagLoader": "Cached Load LoRA Tag",
    # AlcheminePack/API ##############################################################
    "LoadWorkflow": "Load Workflow",
    "ApiGenerate": "Api Generate",
    "ApiSubmit": "Api Submit",
    "ApiCollect": "Api Collect",
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
    "ConsistencyGuard": "Consistency Guard",
    "ClassifyTags": "Classify Tags",
}
