"""Custom nodes mappings."""

from .nodes.everywhere import AnythingEverywhereExtended
from .nodes.flow_control import LazyExecution
from .nodes.image import AdjustImage
from .nodes.lora import DownloadImage, SaveImageWithText
from .nodes.model import CachedLoraTagLoader
from .nodes.prompt import (
    ProcessTags,
    FilterTags,
    FilterSubtags,
    ReplaceUnderscores,
    FixBreakAfterTIPO,
    SDXLTokenAnalyzer,
    RemoveWeights,
    FilterColors,
    FilterPlurals,
    BoySubjectFilter,
    SDXLAutoBreak,
    SubstituteTags,
    SeparateLoraTags,
    TextPrompt,
)


NODE_CLASS_MAPPINGS = {
    # AlcheminePack/Everywhere #######################################################
    "Anything Everywhere Extended": AnythingEverywhereExtended,
    # AlcheminePack/FlowControl ######################################################
    "LazyExecution": LazyExecution,
    # AlcheminePack/Image ############################################################
    "AdjustImage": AdjustImage,
    # AlcheminePack/Lora #############################################################
    "DownloadImage": DownloadImage,
    "SaveImageWithText": SaveImageWithText,
    # AlcheminePack/Model ############################################################
    "CachedLoraTagLoader": CachedLoraTagLoader,
    # AlcheminePack/Prompt #############################################################
    "ProcessTags": ProcessTags,
    "FilterTags": FilterTags,
    "FilterSubtags": FilterSubtags,
    "ReplaceUnderscores": ReplaceUnderscores,
    "FixBreakAfterTIPO": FixBreakAfterTIPO,
    "SDXLTokenAnalyzer": SDXLTokenAnalyzer,
    "RemoveWeights": RemoveWeights,
    "FilterColors": FilterColors,
    "FilterPlurals": FilterPlurals,
    "BoySubjectFilter": BoySubjectFilter,
    "SDXLAutoBreak": SDXLAutoBreak,
    "SubstituteTags": SubstituteTags,
    "SeparateLoraTags": SeparateLoraTags,
    "TextPrompt": TextPrompt,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # AlcheminePack/Everywhere #######################################################
    "Anything Everywhere Extended": "Everywhere",
    # AlcheminePack/FlowControl ######################################################
    "LazyExecution": "Lazy Execution",
    # AlcheminePack/Image ############################################################
    "AdjustImage": "Adjust Image",
    # AlcheminePack/Lora #############################################################
    "DownloadImage": "Download Image",
    "SaveImageWithText": "Save Image With Text",
    # AlcheminePack/Model ############################################################
    "CachedLoraTagLoader": "Cached Load LoRA Tag",
    # AlcheminePack/Prompt #############################################################
    "ProcessTags": "Process Tags",
    "FilterTags": "Filter Tags",
    "FilterSubtags": "Filter Subtags",
    "ReplaceUnderscores": "Replace Underscores",
    "FixBreakAfterTIPO": "Fix Break After TIPO",
    "SDXLTokenAnalyzer": "SDXL Token Analyzer",
    "RemoveWeights": "Remove Weights",
    "FilterColors": "Filter Colors",
    "FilterPlurals": "Filter Plurals",
    "BoySubjectFilter": "Boy Subject Filter",
    "SDXLAutoBreak": "SDXL Auto Break",
    "SubstituteTags": "Substitute Tags",
    "SeparateLoraTags": "Separate Lora Tags",
    "TextPrompt": "Text Prompt",
}


WEB_DIRECTORY = "./web/js"
