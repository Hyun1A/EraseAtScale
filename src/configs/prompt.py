from typing import Literal, Optional, Union

import yaml
from pathlib import Path
import pandas as pd
import random
import numpy as np

from pydantic import BaseModel, root_validator
from transformers import CLIPTextModel, CLIPTokenizer
import torch

from src.misc.clip_templates import imagenet_templates
from src.engine.train_util import encode_prompts

from .prompt_util import smooth_tensor

ACTION_TYPES = Literal[
    "erase",
    "erase_with_la",
]

class PromptEmbedsXL:
    text_embeds: torch.FloatTensor
    pooled_embeds: torch.FloatTensor

    def __init__(self, embeds) -> None:
        self.text_embeds, self.pooled_embeds = embeds

PROMPT_EMBEDDING = Union[torch.FloatTensor, PromptEmbedsXL]


class PromptEmbedsCache:
    
    prompts = {}

    def __setitem__(self, __name, __value):
        self.prompts[__name] = __value

    def __getitem__(self, __name: str):
        if __name in self.prompts:
            return self.prompts[__name]
        else:
            return None


class PromptSettings(BaseModel):  # yaml
    target: Union[str, list]
    positive: Union[str, list] = None  # if None, target will be used
    unconditional: str = ""  # default is ""
    neutral: str = None  # if None, unconditional will be used
    action: ACTION_TYPES = "erase"  # default is "erase"
    guidance_scale: float = 1.0  # default is 1.0
    resolution: int = 512  # default is 512
    dynamic_resolution: bool = False  # default is False
    batch_size: int = 1  # default is 1
    dynamic_crops: bool = False  # default is False. only used when model is XL
    use_template: bool = False  # default is False
    
    la_strength: float = 1000.0
    sampling_batch_size: int = 4

    seed: int = None
    case_number: int = 0

    @root_validator(pre=True)
    def fill_prompts(cls, values):
        keys = values.keys()
        if "target" not in keys:
            raise ValueError("target must be specified")
        if "positive" not in keys:
            values["positive"] = values["target"]
        if "unconditional" not in keys:
            values["unconditional"] = ""
        if "neutral" not in keys:
            values["neutral"] = values["unconditional"]

        return values


class PromptEmbedsPair:
    target: PROMPT_EMBEDDING  # the concept that do not want to generate 
    positive: PROMPT_EMBEDDING  # generate the concept
    unconditional: PROMPT_EMBEDDING  # uncondition (default should be empty)
    neutral: PROMPT_EMBEDDING  # base condition (default should be empty)
    use_template: bool = False  # use clip template or not

    guidance_scale: float
    resolution: int
    dynamic_resolution: bool
    batch_size: int
    dynamic_crops: bool

    loss_fn: torch.nn.Module
    action: ACTION_TYPES

    def __init__(
        self,
        loss_fn: torch.nn.Module,
        target: PROMPT_EMBEDDING,
        positive: PROMPT_EMBEDDING,
        unconditional: PROMPT_EMBEDDING,
        neutral: PROMPT_EMBEDDING,
        settings: PromptSettings,
    ) -> None:
        self.loss_fn = loss_fn
        self.target = target
        self.positive = positive
        self.unconditional = unconditional
        self.neutral = neutral
        
        self.settings = settings

        self.use_template = settings.use_template
        self.guidance_scale = settings.guidance_scale
        self.resolution = settings.resolution
        self.dynamic_resolution = settings.dynamic_resolution
        self.batch_size = settings.batch_size
        self.dynamic_crops = settings.dynamic_crops
        self.action = settings.action
        
        self.la_strength = settings.la_strength
        self.sampling_batch_size = settings.sampling_batch_size
        
        

def load_prompts_from_yaml(path) -> list[PromptSettings]:
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)

    if len(prompts) == 0:
        raise ValueError("prompts file is empty")

    prompt_settings = [PromptSettings(**prompt) for prompt in prompts]

    return prompt_settings



def load_prompts(path) -> list[PromptSettings]:
    df = pd.read_csv(path)

    fields = [k for k in df.keys()]

    if 'actor' in fields:
        prompts = df['actor'].tolist()
    elif 'character' in fields:
        prompts = df['character'].tolist()
    elif 'style' in fields:
        prompts = df['style'].tolist()

    return prompts



def load_prompts_from_table(path) -> list[PromptSettings]:
    # check if the file ends with .csv
    if not path.endswith(".csv"):
        raise ValueError("prompts file must be a csv file")
    df = pd.read_csv(path)
    prompt_settings = []
    for _, row in df.iterrows():
        prompt_settings.append(PromptSettings(**dict(
            target=str(row.prompt),
            seed=int(row.get('sd_seed', row.evaluation_seed)),
            case_number=int(row.get('case_number', -1)),
        )))
    return prompt_settings

def compute_rotation_matrix(target: torch.FloatTensor):
    """Compute the matrix that rotate unit vector to target.
    
    Args:
        target (torch.FloatTensor): target vector.
    """
    normed_target = target.view(-1) / torch.norm(target.view(-1), p=2)
    n = normed_target.shape[0]
    basis = torch.eye(n).to(target.device)
    basis[0] = normed_target
    for i in range(1, n):
        w = basis[i]
        for j in range(i):
            w = w - torch.dot(basis[i], basis[j]) * basis[j]
        basis[i] = w / torch.norm(w, p=2)
    return torch.linalg.inv(basis)