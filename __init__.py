"""Custom nodes mappings."""

from .nodes.everywhere import AnythingEverywhereExtended
from .nodes.flow_control import LazyExecution
from .nodes.grok import GrokGenerate, GrokSubmit, GrokCollect
from .nodes.image import AdjustImage
from .nodes.inference import OpenAIInference
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
    # AlcheminePack/Grok #############################################################
    "GrokGenerate": GrokGenerate,
    "GrokSubmit": GrokSubmit,
    "GrokCollect": GrokCollect,
    # AlcheminePack/Image ############################################################
    "AdjustImage": AdjustImage,
    # AlcheminePack/Inference ########################################################
    "OpenAIInference": OpenAIInference,
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
    # AlcheminePack/Grok #############################################################
    "GrokGenerate": "Grok Generate",
    "GrokSubmit": "Grok Submit",
    "GrokCollect": "Grok Collect",
    # AlcheminePack/Image ############################################################
    "AdjustImage": "Adjust Image",
    # AlcheminePack/Inference ########################################################
    "OpenAIInference": "OpenAI Inference",
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
    "BoySubjectFilter": "Boy Subject Filter",
    "SDXLAutoBreak": "SDXL Auto Break",
    "SubstituteTags": "Substitute Tags",
    "SeparateLoraTags": "Separate Lora Tags",
    "TextPrompt": "Text Prompt",
}


WEB_DIRECTORY = "./web/js"
