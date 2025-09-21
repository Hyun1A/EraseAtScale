# ref:
# - https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# - https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py
# - https://github.com/Con6924/SPM


import os
from copy import deepcopy
import math
from typing import Optional, List, Iterable, Union, Literal
import numpy as np
from src.models.merge_eas import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file
from contextlib import contextmanager
import hashlib

from .eas_layers import EASLayer_Gate, Attention_Gate

        
        
class EASNetwork(nn.Module):
    UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
        "Transformer2DModel",
    ]
    UNET_TARGET_REPLACE_MODULE_CONV = [
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
    ]

    EAS_PREFIX_UNET = "lora_unet"   # aligning with SD webui usage
    DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

    def __init__(
        self,
        unet: UNet2DConditionModel,
        text_encoder: CLIPTextModel,
        rank: int = 4,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        module = EASLayer_Gate,
        module_kwargs = None,
        delta=1e-5,
        num_embeddings=77,
        text_emb_dimension=768,
        hidden_size=4,
        edit_scale=2.5,
        continual=False,
        task_id=None,    
        continual_rank=4,
        init_size=4,
        n_concepts=1,
        args=None,
    ) -> None:
        
        super().__init__()
        
        self.continual=continual
        self.task_id=task_id
        self.continual_rank=continual_rank
        self.n_concepts = n_concepts
        
        self.multiplier = multiplier
        self.dim = rank
        self.alpha = alpha
        self.delta = delta 
        
        self.module = module
        self.module_kwargs = module_kwargs or {}
        
        self.num_embeddings = num_embeddings
        self.text_emb_dimension = text_emb_dimension
        self.hidden_size = hidden_size
        self.init_size = init_size
        self.edit_scale = edit_scale
        self.args = args

        ####################################################
        ####################################################

        self.attention_gate = Attention_Gate(input_size=self.text_emb_dimension, init_size=self.init_size,\
                                  hidden_size=self.hidden_size, \
                                  num_embeddings=self.num_embeddings, task_id=self.task_id, n_concepts=self.n_concepts)


        
        
        self.unet_eas_layers = self.create_modules(
            EASNetwork.EAS_PREFIX_UNET,
            unet,
            EASNetwork.DEFAULT_TARGET_REPLACE,
            self.dim,
            self.multiplier,
        )

        print(f"Create EAS for U-Net: {len(self.unet_eas_layers)} modules.")

        eas_names = set()
        for eas_layer in self.unet_eas_layers:
            assert (
                eas_layer.eas_name not in eas_names
            ), f"duplicated EAS layer name: {eas_layer.eas_name}. {eas_names}"
            eas_names.add(eas_layer.eas_name)

        ############### Add: printing modified text encoder module ################
        for eas_layer in self.unet_eas_layers:
            eas_layer.apply_to()
            self.add_module(
                eas_layer.eas_name,
                eas_layer,
            )
        
        del unet
        
                
    def reset_cache_attention_gate(self):
        for layer in self.unet_eas_layers:
            layer.att_output = None
      
    def set_inference_mode(self):
        for layer in self.unet_eas_layers:
            layer.inference_mode = True    
      
    def set_train_mode(self):
        for layer in self.unet_eas_layers:
            layer.inference_mode = False        
        

    
    def load_eas_lora_models(self, model_paths):
        for layer in self.unet_eas_layers:
            self.attention.encoder_layer.add_slf_attn(model_paths, layer.eas_name)
        

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
    ) -> list:
        eas_layers = []

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d"]:
                        if not ("attn2" in child_name):
                            continue
                        
                        if self.args.net_type == "ca_kv":
                            if not(("to_k" in child_name) or ("to_v" in child_name)):
                                continue
                        if self.args.net_type == "ca_v":
                            if not "to_v" in child_name:
                                continue
                        if self.args.net_type == "ca_sel_kv":
                            if not (("up_blocks.2.attentions.0" in name) or ("up_blocks.1.attentions.2" in name)):
                                continue
                            if not(("to_k" in child_name) or ("to_v" in child_name)):
                                continue                            
                        if self.args.net_type == "ca_sel_block":
                            if not (("up_blocks.1" in name) or ("up_blocks.2" in name) or ("down_blocks.2" in name)):
                                continue
                            if not(("to_k" in child_name) or ("to_v" in child_name)):
                                continue
                            
                        eas_name = prefix + "." + name + "." + child_name
                        eas_name = eas_name.replace(".", "_")
                        print(f"{eas_name}")
                        
                        
                        eas_layer = self.module(
                            eas_name, child_module, multiplier, rank, self.alpha, \
                            init_size=self.init_size, hidden_size=self.hidden_size, \
                            num_embeddings=self.num_embeddings, \
                            task_id=self.task_id, \
                            # attention_gate=self.attention_gate, \
                            attention_gate=self.attention_gate, \
                            n_concepts=self.n_concepts, args=self.args, \
                            **self.module_kwargs
                        )
                        eas_layers.append(eas_layer)

        return eas_layers
    
    
    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):    
        all_params = []

        param_data = {"params": self.parameters()}
        if default_lr is not None:
            param_data["lr"] = default_lr
        all_params.append(param_data)                

        return all_params             
    
    
    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        
        state_dict = self.state_dict()
        
        state_dict_save = dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                if ("lora" in key) and ("attention_gate" in key):
                    continue                
                
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict_save[key] = v
                
        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict_save, file, metadata)
        else:
            torch.save(state_dict_save, file)


    
    def __enter__(self):
        for eas_layer in self.unet_eas_layers:
            eas_layer.multiplier = 1.0
            eas_layer.use_prompt_tuning = True

            
    def __exit__(self, exc_type, exc_value, tb):
        for eas_layer in self.unet_eas_layers:
            eas_layer.multiplier = 0
            eas_layer.use_prompt_tuning = False
                        

    @contextmanager
    def editing(self):
        try:
            for c in self.unet_eas_layers:
                c.multiplier = 1.0
                c.use_prompt_tuning = True
            yield self
        finally:
            for c in self.unet_eas_layers:
                c.multiplier = 0
                c.use_prompt_tuning = False
