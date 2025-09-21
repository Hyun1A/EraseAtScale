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


class ParamModule(nn.Module):
    def __init__(self, size):
        super(ParamModule, self).__init__()
        self.weight = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return x * self.weight

    def __repr__(self):
        return f"ParameterModule(param_shape={tuple(self.weight.shape)})"

    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q_list, k_list, mask=None):
        
        
        for i, (q, k) in enumerate(zip(q_list, k_list)):
            if i == 0:
                attn = torch.matmul(q, k.transpose(3, 4)) / self.temperature
            else:
                attn += torch.matmul(q, k.transpose(3, 4)) / self.temperature
        
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.sum(dim=2)
        attn = F.softmax(attn, dim=-1)

        
        return attn
        
        
class AttentionModule(nn.Module):
    def __init__(self, init_size, n_head, d_model, d_k, dropout=0.25, task_id=0, n_concepts=1):
        super().__init__()        
        
        self.n_head = n_head
        self.d_model = d_model
        self.init_size = init_size
        self.n_concepts = n_concepts
        self.temperature = 1.0
        
        self.d_k = n_head * d_k
        self.task_id = task_id
        
        self.w_qs = ParamModule((n_concepts, d_model, init_size))
        self.w_ks = ParamModule((n_concepts, d_model, init_size))
        
        self.w_qs_list = [self.w_qs]
        self.w_ks_list = [self.w_ks]
        
        nn.init.kaiming_uniform_(self.w_qs.weight, a=math.sqrt(5))
        self.w_qs.weight.data = self.w_qs.weight.data / (d_model**2)    
        nn.init.kaiming_uniform_(self.w_ks.weight, a=math.sqrt(5))
        self.w_ks.weight.data = self.w_ks.weight.data / (d_model**2)          

        
        self.attention = ScaledDotProductAttention(temperature=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        q,k,v = x,x,x
        
        input = q.unsqueeze(1)

        d_k, _, n_head = 1, 1, self.n_head
        sz_b, len_q, len_k, _ = q.size(0), q.size(1), k.size(1), v.size(1)        
        
        q_list = []
        k_list = []
                            
        for w_qs, w_ks in zip(self.w_qs_list, self.w_ks_list):
            wq = torch.einsum("btd,ndh->bnth", q, w_qs.weight)  # 10x50x77x768, 50x768x4
            wk = torch.einsum("btd,ndh->bnth", k, w_ks.weight)
            
            q_list.append(wq.view(sz_b, self.n_concepts, len_q, n_head, wq.shape[-1]//n_head).transpose(2, 3))
            k_list.append(wq.view(sz_b, self.n_concepts, len_k, n_head, wk.shape[-1]//n_head).transpose(2, 3))
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        attn = self.attention(q_list, k_list, mask=mask)
        
        e_aggregated = torch.einsum("bnst,bitd->bnsd", attn, input)
        
        return e_aggregated, attn    
    

    
class GateModule(nn.Module):
    def __init__(self, init_size, n_head, d_model, d_k, dropout=0.25, task_id=0, n_concepts=1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.init_size = init_size
        self.n_concepts = n_concepts
        self.temperature = 1.0
        
        self.d_k = n_head * d_k
        self.task_id = task_id
        
        self.scaler = ParamModule((n_concepts, d_model, 1))      
        self.bias_gate = nn.Parameter(torch.zeros(n_concepts, d_model)) # (C,d_model)
        
        nn.init.zeros_(self.scaler.weight)
        

    def forward(self, e_aggregated):

        feat = torch.einsum("ndi,bntd->bnti", self.scaler.weight, (e_aggregated+self.bias_gate[None,:,None,:]))
        scale = torch.sigmoid(self.temperature * feat)
        scale_max_val, scale_max_ind = scale.max(dim=1, keepdim=True)
        # breakpoint()
        ind_one_hot = F.one_hot(scale_max_ind.squeeze(-1), num_classes=self.n_concepts).permute(0,3,2,1)
        e_aggregated = (scale_max_val*ind_one_hot*e_aggregated).sum(dim=1)
        
        return e_aggregated, ind_one_hot, scale_max_val, scale_max_ind
    

class MultiHeadAttention(nn.Module):
    
    def __init__(self, init_size, n_head, d_model, d_k, dropout=0.25, task_id=0, n_concepts=1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.init_size = init_size
        self.n_concepts = n_concepts
        self.temperature = 1.0
        
        self.d_k = n_head * d_k
        self.task_id = task_id 

        self.attention_module1 = AttentionModule(init_size, n_head, d_model, d_k, n_concepts=n_concepts)
        
        self.gate = GateModule(init_size, n_head, d_model, d_k, n_concepts=n_concepts)
        
    def forward(self, x, mask=None):
        e_aggregated = x.unsqueeze(dim=1).repeat(1,self.n_concepts,1,1)
        e_aggregated, ind_one_hot, _, _ = self.gate(e_aggregated)
        
        return e_aggregated, None, ind_one_hot    
    

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, init_size, d_k, d_v, dropout=0.1, task_id=0, n_concepts=1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(init_size, n_head, d_model, d_k, dropout=dropout, task_id=task_id, n_concepts=n_concepts)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn, ind_one_hot = self.slf_attn(
            enc_input, mask=slf_attn_mask)
        return enc_output, enc_slf_attn, ind_one_hot
    
    
    
class Attention_Gate(nn.Module):
    def __init__(self, input_size=768, init_size=16, hidden_size=16, num_embeddings=77, n_head=1, dropout=0.5, task_id=0, n_concepts=1):
        super(Attention_Gate, self).__init__()   
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init_size = init_size
        self.n_head = n_head
        self.num_embeddings = num_embeddings
        self.inner_size = self.n_head * self.hidden_size
        self.enc_cache = None
        self.last_prompt = None
        
        self.encoder_layer = EncoderLayer(self.input_size, self.inner_size, \
                  self.n_head, self.init_size, self.hidden_size, self.hidden_size, dropout=dropout, task_id=task_id, n_concepts=n_concepts)
        
        self.use_cache = False
    
        self.enc_output = None
        self.ind_one_hot = None
        
    def forward(self, x):
        enc_output, _, ind_one_hot  = self.encoder_layer(x, slf_attn_mask=None)
        # breakpoint()

        return enc_output, ind_one_hot         

        if not self.use_cache:
            enc_output, _, ind_one_hot  = self.encoder_layer(x, slf_attn_mask=None)
            return enc_output, ind_one_hot 
        
        elif (self.use_cache) and (self.enc_output is None):
            enc_output, _, ind_one_hot  = self.encoder_layer(x, slf_attn_mask=None)
            self.enc_output = enc_output
            self.ind_one_hot = ind_one_hot
            return self.enc_output, self.ind_one_hot    
        
        else:
            return self.enc_output, self.ind_one_hot     

    def reset_cache(self):
        self.enc_output = None
        self.ind_one_hot = None
    
    def set_cache(self, flag):
        self.reset_cache()
        self.use_cache = flag
        
    
    
class EASLayer_Gate(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the otrain_prompt_editor_mixup_sampriginal Linear module.
    """

    def __init__(
        self,
        eas_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
        init_size=1,
        hidden_size=1,
        num_embeddings=77,
        task_id=1,
        attention_gate=None,
        n_concepts=1,
        args=None,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.eas_name = eas_name
        self.dim = dim

        if org_module.__class__.__name__ == "Linear":
            in_dim = org_module.in_features
            out_dim = org_module.out_features

            # dim of lora_down: N x D x H
            # dim of lora_up: N x H x D          
            self.lora_down = ParamModule((n_concepts, in_dim, dim))
            self.lora_up = ParamModule((n_concepts, dim, out_dim))  
            

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)            
            
        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

        self.attention_gate = attention_gate

        self.use_prompt_tuning = False    
        
        self.inference_mode = False
        self.att_output = None        
        
    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        # x.shape: (B, 77, 768)
                
        if self.use_prompt_tuning:    
            if not self.inference_mode:
                edit_direction, ind_one_hot = self.attention_gate(x)

                # dim of edit_direction: B x T x D
                # dim of ind_one_hot: B x N x T x 1
                # dim of lora_down: N x D x H
                # dim of lora_up: N x H x D

                # selection_down: (lora_down)NDH, (ind_one_hot)BNT1 -> BTDH
                # selection_up  : (lora_up)NHD, (ind_one_hot)BNT1 -> BTHD
                # compute down  : (edit)BTD, (selected_lora_down)BTDH -> BTH 
                # compute up    : (edit)BTH, (selected_lora_down)BTHD -> BTD 
                selection_down = torch.einsum("ndh,bnti->btdh", self.lora_down.weight, ind_one_hot.float())
                selection_up = torch.einsum("nhd,bnti->bthd", self.lora_up.weight, ind_one_hot.float())
                down = torch.einsum("btd,btdh->bth", edit_direction, selection_down)
                up = torch.einsum("bth,bthd->btd", down, selection_up)

                return (
                    self.org_forward(x) + up * self.multiplier * self.scale                
                )

            elif (self.inference_mode) and (self.att_output is None):
                edit_direction, ind_one_hot = self.attention_gate(x)
                selection_down = torch.einsum("ndh,bnti->btdh", self.lora_down.weight, ind_one_hot.float())
                selection_up = torch.einsum("nhd,bnti->bthd", self.lora_up.weight, ind_one_hot.float())
                down = torch.einsum("btd,btdh->bth", edit_direction, selection_down)
                up = torch.einsum("bth,bthd->btd", down, selection_up)
                self.att_output = up
                    
                return (
                    self.org_forward(x) + self.att_output * self.multiplier * self.scale                
                )
                    
            else:
                return (
                    self.org_forward(x) + self.att_output * self.multiplier * self.scale                
                )
        
            
        else:
            return (
                self.org_forward(x)
            )        


        



ActivationName = Literal[
    "relu", "leaky_relu", "prelu",
    "elu", "selu", "gelu", "silu", "swish", "mish",
    "tanh", "sigmoid", "identity"
]

def activation_factory(name: ActivationName, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=kwargs.get("inplace", True))
    if name == "leaky_relu":
        return nn.LeakyReLU(
            negative_slope=kwargs.get("negative_slope", 0.01),
            inplace=kwargs.get("inplace", True),
        )
    if name == "prelu":
        return nn.PReLU(
            num_parameters=kwargs.get("num_parameters", 1),
            init=kwargs.get("init", 0.25),
        )
    if name == "elu":
        return nn.ELU(alpha=kwargs.get("alpha", 1.0), inplace=kwargs.get("inplace", True))
    if name == "selu":
        return nn.SELU(inplace=kwargs.get("inplace", True))
    if name == "gelu":
        return nn.GELU(approximate=kwargs.get("approximate", "none"))
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=kwargs.get("inplace", True))
    if name == "mish":
        return nn.Mish()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")



class EASLayer_MLP(nn.Module):
    def __init__(
        self,
        eas_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,                 
        alpha=1,
        init_size=1,          
        hidden_size=256,
        num_embeddings=77,    
        task_id=1,   
        attention_gate=None,
        n_concepts=1,
        args=None,
    ):
        super().__init__()
        self.eas_name = eas_name
        self.dim = dim
        self.depth = max(1, int(getattr(args, "depth", 1)))
        self.hidden_size = int(hidden_size)
        self.n_concepts = int(n_concepts)
        self.activation = args.activation
        self.use_bias = args.use_bias

        self.in_dim = org_module.in_features
        self.out_dim = org_module.out_features

        # ----- per-concept MLP -----
        self.w1 = nn.Parameter(torch.empty(self.n_concepts, self.in_dim, self.dim))
        self.b1 = nn.Parameter(torch.zeros(self.n_concepts, self.dim))

        self.mid_w = nn.ParameterList()
        self.mid_b = nn.ParameterList()
        for _ in range(max(0, self.depth - 2)):
            self.mid_w.append(nn.Parameter(torch.empty(self.n_concepts, self.dim, self.dim)))
            self.mid_b.append(nn.Parameter(torch.zeros(self.n_concepts, self.dim)))

        self.w_out = nn.Parameter(torch.empty(self.n_concepts, self.dim, self.out_dim))
        self.b_out = nn.Parameter(torch.zeros(self.n_concepts, self.out_dim))


        self.activation = activation_factory(args.activation)


        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        for w in self.mid_w:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        nn.init.zeros_(self.w_out)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().cpu().numpy().item()
        alpha = self.dim if alpha is None or alpha == 0 else alpha
        self.scale = float(alpha) / float(self.dim)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        self.multiplier = multiplier
        self.org_module = org_module

        self.use_prompt_tuning = False

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _forward_all_concepts_mlp(self, x):
        B, T, D = x.shape

        if self.use_bias:
            h = torch.einsum("btd,ndh->btnh", x, self.w1) + self.b1.view(1, 1, self.n_concepts, self.dim)
        else:
            h = torch.einsum("btd,ndh->btnh", x, self.w1) 
        h = self.activation(h)

        for w_m, b_m in zip(self.mid_w, self.mid_b):
            if self.use_bias:
                h = torch.einsum("btnh,nhh->btnh", h, w_m) + b_m.view(1, 1, self.n_concepts, self.dim)
            else:
                h = torch.einsum("btnh,nhh->btnh", h, w_m)
            h = self.activation(h)

        y_all = torch.einsum("btnh,nhd->btnd", h, self.w_out) + self.b_out.view(1, 1, self.n_concepts, self.out_dim)

        y = y_all.sum(dim=2)  # (B,T,D_out)
        return y

    def forward(self, x):
        # x: (B, T, in_dim)
        if not self.use_prompt_tuning:
            return self.org_forward(x)

        up = self._forward_all_concepts_mlp(x)  # (B,T,out_dim)
        return self.org_forward(x) + up * self.multiplier * self.scale








class EASLayer_MLP_SwiGLU(nn.Module):
    def __init__(
        self,
        eas_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
        init_size=1,
        hidden_size=256,
        num_embeddings=77,
        task_id=1,
        attention_gate=None,
        n_concepts=1,
        args=None,
    ):
        super().__init__()
        self.eas_name = eas_name
        self.dim = int(dim)
        self.depth = max(1, int(getattr(args, "depth", 1)))
        self.hidden_size = int(hidden_size)
        self.n_concepts = int(n_concepts)
        self.use_bias = bool(getattr(args, "use_bias", True))

        self.in_dim = org_module.in_features
        self.out_dim = org_module.out_features

        # ----- per-concept SwiGLU MLP -----
        self.w1_a = nn.Parameter(torch.empty(self.n_concepts, self.in_dim, self.dim))
        self.w1_b = nn.Parameter(torch.empty(self.n_concepts, self.in_dim, self.dim))
        if self.use_bias:
            self.b1_a = nn.Parameter(torch.zeros(self.n_concepts, self.dim))
            self.b1_b = nn.Parameter(torch.zeros(self.n_concepts, self.dim))
        else:
            self.register_parameter("b1_a", None)
            self.register_parameter("b1_b", None)

        self.mid_w_a = nn.ParameterList()
        self.mid_w_b = nn.ParameterList()
        self.mid_b_a = nn.ParameterList()
        self.mid_b_b = nn.ParameterList()
        for _ in range(max(0, self.depth - 2)):
            self.mid_w_a.append(nn.Parameter(torch.empty(self.n_concepts, self.dim, self.dim)))
            self.mid_w_b.append(nn.Parameter(torch.empty(self.n_concepts, self.dim, self.dim)))
            if self.use_bias:
                self.mid_b_a.append(nn.Parameter(torch.zeros(self.n_concepts, self.dim)))
                self.mid_b_b.append(nn.Parameter(torch.zeros(self.n_concepts, self.dim)))
            else:
                self.mid_b_a.append(None)
                self.mid_b_b.append(None)

        self.w_out = nn.Parameter(torch.empty(self.n_concepts, self.dim, self.out_dim))
        self.b_out = nn.Parameter(torch.zeros(self.n_concepts, self.out_dim))

        # ----- init -----
        nn.init.kaiming_uniform_(self.w1_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w1_b, a=math.sqrt(5))
        for wa, wb in zip(self.mid_w_a, self.mid_w_b):
            nn.init.kaiming_uniform_(wa, a=math.sqrt(5))
            nn.init.kaiming_uniform_(wb, a=math.sqrt(5))
        nn.init.zeros_(self.w_out)

        # scale/alpha
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().cpu().numpy().item()
        alpha = self.dim if alpha is None or alpha == 0 else alpha
        self.scale = float(alpha) / float(self.dim)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        self.multiplier = multiplier
        self.org_module = org_module
        self.use_prompt_tuning = False

        self.silu = nn.SiLU(inplace=True)

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    # ----- forward helpers -----
    def _forward_all_concepts_mlp_swiglu(self, x):
        # x: (B, T, in_dim)
        if self.use_bias and (self.b1_a is not None):
            h_a = torch.einsum("btd,ndh->btnh", x, self.w1_a) + self.b1_a.view(1, 1, self.n_concepts, self.dim)
            h_b = torch.einsum("btd,ndh->btnh", x, self.w1_b) + self.b1_b.view(1, 1, self.n_concepts, self.dim)
        else:
            h_a = torch.einsum("btd,ndh->btnh", x, self.w1_a)
            h_b = torch.einsum("btd,ndh->btnh", x, self.w1_b)
        h = self.silu(h_a) * h_b  # SwiGLU

        for w_a, w_b, b_a, b_b in zip(self.mid_w_a, self.mid_w_b, self.mid_b_a, self.mid_b_b):
            if self.use_bias and (b_a is not None):
                h_a = torch.einsum("btnh,nhh->btnh", h, w_a) + b_a.view(1, 1, self.n_concepts, self.dim)
                h_b = torch.einsum("btnh,nhh->btnh", h, w_b) + b_b.view(1, 1, self.n_concepts, self.dim)
            else:
                h_a = torch.einsum("btnh,nhh->btnh", h, w_a)
                h_b = torch.einsum("btnh,nhh->btnh", h, w_b)
            h = self.silu(h_a) * h_b

        y_all = torch.einsum("btnh,nhd->btnd", h, self.w_out) + self.b_out.view(1, 1, self.n_concepts, self.out_dim)
        y = y_all.sum(dim=2)  # (B, T, out_dim)
        return y

    def forward(self, x):
        # x: (B, T, in_dim)
        if not self.use_prompt_tuning:
            return self.org_forward(x)
        up = self._forward_all_concepts_mlp_swiglu(x)
        return self.org_forward(x) + up * self.multiplier * self.scale







class EASLayer_Linear(nn.Module):
    def __init__(
        self,
        eas_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
        init_size=1,
        hidden_size=256,  
        num_embeddings=77, 
        task_id=1,         
        attention_gate=None,
        n_concepts=1,
        args=None,
    ):
        super().__init__()
        self.eas_name = eas_name
        self.dim = int(dim)
        self.depth = max(1, int(getattr(args, "depth", 1)))
        self.n_concepts = int(n_concepts)
        self.use_bias = bool(getattr(args, "use_bias", True))

        self.in_dim = org_module.in_features
        self.out_dim = org_module.out_features

        # ----- per-concept Linear stack -----
        self.w1 = nn.Parameter(torch.empty(self.n_concepts, self.in_dim, self.dim))
        self.b1 = nn.Parameter(torch.zeros(self.n_concepts, self.dim)) if self.use_bias else None

        self.mid_w = nn.ParameterList()
        self.mid_b = nn.ParameterList()
        for _ in range(max(0, self.depth - 2)):
            self.mid_w.append(nn.Parameter(torch.empty(self.n_concepts, self.dim, self.dim)))
            self.mid_b.append(nn.Parameter(torch.zeros(self.n_concepts, self.dim)) if self.use_bias else None)

        # hidden -> out
        self.w_out = nn.Parameter(torch.empty(self.n_concepts, self.dim, self.out_dim))
        self.b_out = nn.Parameter(torch.zeros(self.n_concepts, self.out_dim))

        # ----- init -----
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        for w in self.mid_w:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        nn.init.zeros_(self.w_out)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().cpu().numpy().item()
        alpha = self.dim if alpha is None or alpha == 0 else alpha
        self.scale = float(alpha) / float(self.dim)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        self.multiplier = multiplier
        self.org_module = org_module
        self.use_prompt_tuning = False

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _forward_all_concepts_linear(self, x):
        # x: (B, T, in_dim)
        if self.use_bias and self.b1 is not None:
            h = torch.einsum("btd,ndh->btnh", x, self.w1) + self.b1.view(1, 1, self.n_concepts, self.dim)
        else:
            h = torch.einsum("btd,ndh->btnh", x, self.w1)

        for w_m, b_m in zip(self.mid_w, self.mid_b):
            if self.use_bias and (b_m is not None):
                h = torch.einsum("btnh,nhh->btnh", h, w_m) + b_m.view(1, 1, self.n_concepts, self.dim)
            else:
                h = torch.einsum("btnh,nhh->btnh", h, w_m)

        y_all = torch.einsum("btnh,nhd->btnd", h, self.w_out) + self.b_out.view(1, 1, self.n_concepts, self.out_dim)
        y = y_all.sum(dim=2)  # (B, T, out_dim)
        return y

    def forward(self, x):
        # x: (B, T, in_dim)
        if not self.use_prompt_tuning:
            return self.org_forward(x)
        up = self._forward_all_concepts_linear(x)
        return self.org_forward(x) + up * self.multiplier * self.scale



        
# ============ Norm helpers ============
class SharedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight

class SharedScaleNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(1.0))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return self.g * x / norm

def make_shared_norm(norm: str | None, dim: int) -> nn.Module:
    if norm is None or str(norm).lower() in {"none", ""}:
        return nn.Identity()
    n = str(norm).lower()
    if n == "layernorm":
        return nn.LayerNorm(dim)
    if n == "rmsnorm":
        return SharedRMSNorm(dim)
    if n == "scalenorm":
        return SharedScaleNorm(dim)
    raise ValueError(f"Unknown norm: {norm}")

class PerConceptLayerNorm(nn.Module):
    def __init__(self, n_concepts: int, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_concepts, dim))
        self.bias = nn.Parameter(torch.zeros(n_concepts, dim))
        self.eps = eps
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, N, dim)
        mean = h.mean(dim=-1, keepdim=True)
        var = h.var(dim=-1, unbiased=False, keepdim=True)
        h = (h - mean) / torch.sqrt(var + self.eps)
        return h * self.weight.view(1,1,*self.weight.shape) + self.bias.view(1,1,*self.bias.shape)

class PerConceptRMSNorm(nn.Module):
    """RMSNorm with per-concept weight: (n_concepts, dim)"""
    def __init__(self, n_concepts: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_concepts, dim))
        self.eps = eps
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        var = h.pow(2).mean(dim=-1, keepdim=True)
        h = h * torch.rsqrt(var + self.eps)
        return h * self.weight.view(1,1,*self.weight.shape)

class PerConceptScaleNorm(nn.Module):
    def __init__(self, n_concepts: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n_concepts, dim))
        self.eps = eps
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        norm = h.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return self.g.view(1,1,*self.g.shape) * h / norm

def make_perconcept_norm(norm: str | None, n_concepts: int, dim: int) -> nn.Module:
    if norm is None or str(norm).lower() in {"none", ""}:
        return nn.Identity()
    n = str(norm).lower()
    if n == "layernorm":
        return PerConceptLayerNorm(n_concepts, dim)
    if n == "rmsnorm":
        return PerConceptRMSNorm(n_concepts, dim)
    if n == "scalenorm":
        return PerConceptScaleNorm(n_concepts, dim)
    raise ValueError(f"Unknown norm: {norm}")

class EASLayer_FFN(nn.Module):
    def __init__(
        self,
        eas_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
        init_size=1,
        hidden_size=256,
        num_embeddings=77,
        task_id=1,
        attention_gate=None,
        n_concepts=1,
        args=None,
    ):
        super().__init__()
        self.eas_name = eas_name
        self.dim = int(dim)
        self.depth = max(1, int(getattr(args, "depth", 1)))
        self.hidden_size = int(hidden_size)
        self.n_concepts = int(n_concepts)
        self.use_bias = bool(getattr(args, "use_bias", True))

        self.in_dim = org_module.in_features
        self.out_dim = org_module.out_features

        # ---- norm options (from args) ----
        self.norm_type = str(getattr(args, "ffn_norm", "rmsnorm")).lower()
        self.prenorm = bool(getattr(args, "ffn_prenorm", True))
        self.dropout = nn.Dropout(float(getattr(args, "ffn_dropout", 0.0)))
        self.resid_dropout = nn.Dropout(float(getattr(args, "ffn_resid_dropout", 0.0)))

        # Shared norm for x: (B,T,in_dim)
        self.input_norm = make_shared_norm(self.norm_type, self.in_dim) if self.prenorm else nn.Identity()
        # Per-concept norms for hidden h: (B,T,N,dim)
        n_mids = max(0, self.depth - 1)
        self.mid_norms = nn.ModuleList([make_perconcept_norm(self.norm_type, self.n_concepts, self.dim) for _ in range(n_mids)])

        # ----- per-concept SwiGLU weights -----
        # first layer
        self.w1_a = nn.Parameter(torch.empty(self.n_concepts, self.in_dim, self.dim))
        self.w1_b = nn.Parameter(torch.empty(self.n_concepts, self.in_dim, self.dim))
        if self.use_bias:
            self.b1_a = nn.Parameter(torch.zeros(self.n_concepts, self.dim))
            self.b1_b = nn.Parameter(torch.zeros(self.n_concepts, self.dim))
        else:
            self.register_parameter("b1_a", None)
            self.register_parameter("b1_b", None)

        # mid layers (depth-2)
        self.mid_w_a = nn.ParameterList()
        self.mid_w_b = nn.ParameterList()
        self.mid_b_a = nn.ParameterList()
        self.mid_b_b = nn.ParameterList()
        for _ in range(n_mids):
            self.mid_w_a.append(nn.Parameter(torch.empty(self.n_concepts, self.dim, self.dim)))
            self.mid_w_b.append(nn.Parameter(torch.empty(self.n_concepts, self.dim, self.dim)))
            if self.use_bias:
                self.mid_b_a.append(nn.Parameter(torch.zeros(self.n_concepts, self.dim)))
                self.mid_b_b.append(nn.Parameter(torch.zeros(self.n_concepts, self.dim)))
            else:
                self.mid_b_a.append(None)
                self.mid_b_b.append(None)

        # output proj
        self.w_out = nn.Parameter(torch.empty(self.n_concepts, self.dim, self.out_dim))
        self.b_out = nn.Parameter(torch.zeros(self.n_concepts, self.out_dim))

        # ----- init -----
        nn.init.kaiming_uniform_(self.w1_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w1_b, a=math.sqrt(5))
        for wa, wb in zip(self.mid_w_a, self.mid_w_b):
            nn.init.kaiming_uniform_(wa, a=math.sqrt(5))
            nn.init.kaiming_uniform_(wb, a=math.sqrt(5))
        nn.init.zeros_(self.w_out)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().cpu().numpy().item()
        alpha = self.dim if alpha is None or alpha == 0 else alpha
        self.scale = float(alpha) / float(self.dim)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        self.multiplier = multiplier
        self.org_module = org_module
        self.use_prompt_tuning = False

        self.silu = nn.SiLU(inplace=True)

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    # ---- helpers ----
    def _first_block(self, x):
        # x: (B,T,in_dim)  -> h: (B,T,N,dim)
        if self.prenorm:
            x = self.input_norm(x)
        if self.use_bias and (self.b1_a is not None):
            h_a = torch.einsum("btd,ndh->btnh", x, self.w1_a) + self.b1_a.view(1,1,self.n_concepts,self.dim)
            h_b = torch.einsum("btd,ndh->btnh", x, self.w1_b) + self.b1_b.view(1,1,self.n_concepts,self.dim)
        else:
            h_a = torch.einsum("btd,ndh->btnh", x, self.w1_a)
            h_b = torch.einsum("btd,ndh->btnh", x, self.w1_b)
        h = self.silu(h_a) * h_b
        h = self.dropout(h)
        return h

    def _mid_block(self, h, wi, wj, bi, bj, norm: nn.Module | None):
        # h: (B,T,N,dim) -> (B,T,N,dim)
        if self.prenorm and (norm is not None):
            h = norm(h)
        if self.use_bias and (bi is not None):
            h_a = torch.einsum("btnh,nhh->btnh", h, wi) + bi.view(1,1,self.n_concepts,self.dim)
            h_b = torch.einsum("btnh,nhh->btnh", h, wj) + bj.view(1,1,self.n_concepts,self.dim)
        else:
            h_a = torch.einsum("btnh,nhh->btnh", h, wi)
            h_b = torch.einsum("btnh,nhh->btnh", h, wj)
        h = self.silu(h_a) * h_b
        h = self.dropout(h)
        return h

    def _last_proj(self, h):
        # h: (B,T,N,dim) -> y: (B,T,out_dim)
        y_all = torch.einsum("btnh,nhd->btnd", h, self.w_out) + self.b_out.view(1,1,self.n_concepts,self.out_dim)
        y_all = self.resid_dropout(y_all)
        y = y_all.sum(dim=2)
        return y

    def _forward_all_concepts_mlp_swiglu(self, x):
        h = self._first_block(x)
        # mid blocks
        for k, (wa, wb, ba, bb) in enumerate(zip(self.mid_w_a, self.mid_w_b, self.mid_b_a, self.mid_b_b)):
            h = self._mid_block(h, wa, wb, ba, bb, self.mid_norms[k] if self.prenorm else None)
        y = self._last_proj(h)
        return y

    def forward(self, x):
        # x: (B, T, in_dim)
        if not self.use_prompt_tuning:
            return self.org_forward(x)
        up = self._forward_all_concepts_mlp_swiglu(x)
        return self.org_forward(x) + up * self.multiplier * self.scale



class EASLayer_FFN_GLU(nn.Module):
    def __init__(
        self,
        eas_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
        init_size=1,
        hidden_size=256,
        num_embeddings=77,
        task_id=1,
        attention_gate=None,
        n_concepts=1,
        args=None,
    ):
        super().__init__()
        self.eas_name = eas_name
        self.dim = int(dim)
        self.depth = max(1, int(getattr(args, "depth", 1)))
        self.hidden_size = int(hidden_size)
        self.n_concepts = int(n_concepts)
        self.use_bias = bool(getattr(args, "use_bias", True))

        self.in_dim = org_module.in_features
        self.out_dim = org_module.out_features

        # ---- norm options (from args) ----
        self.norm_type = str(getattr(args, "ffn_norm", "rmsnorm")).lower()
        self.prenorm = bool(getattr(args, "ffn_prenorm", True))
        self.dropout = nn.Dropout(float(getattr(args, "ffn_dropout", 0.0)))
        self.resid_dropout = nn.Dropout(float(getattr(args, "ffn_resid_dropout", 0.0)))

        # Shared norm for x: (B,T,in_dim)
        self.input_norm = make_shared_norm(self.norm_type, self.in_dim) if self.prenorm else nn.Identity()
        # Per-concept norms for hidden h: (B,T,N,dim)
        n_mids = max(0, self.depth - 1)
        self.mid_norms = nn.ModuleList([make_perconcept_norm(self.norm_type, self.n_concepts, self.dim) for _ in range(n_mids)])

        # ----- per-concept SwiGLU weights -----
        # first layer
        self.w1_a = nn.Parameter(torch.empty(self.n_concepts, self.in_dim, self.dim))
        self.w1_b = nn.Parameter(torch.empty(self.n_concepts, self.in_dim, self.dim))
        if self.use_bias:
            self.b1_a = nn.Parameter(torch.zeros(self.n_concepts, self.dim))
            self.b1_b = nn.Parameter(torch.zeros(self.n_concepts, self.dim))
        else:
            self.register_parameter("b1_a", None)
            self.register_parameter("b1_b", None)

        # mid layers (depth-2)
        self.mid_w_a = nn.ParameterList()
        self.mid_w_b = nn.ParameterList()
        self.mid_b_a = nn.ParameterList()
        self.mid_b_b = nn.ParameterList()
        for _ in range(n_mids):
            self.mid_w_a.append(nn.Parameter(torch.empty(self.n_concepts, self.dim, self.dim)))
            self.mid_w_b.append(nn.Parameter(torch.empty(self.n_concepts, self.dim, self.dim)))
            if self.use_bias:
                self.mid_b_a.append(nn.Parameter(torch.zeros(self.n_concepts, self.dim)))
                self.mid_b_b.append(nn.Parameter(torch.zeros(self.n_concepts, self.dim)))
            else:
                self.mid_b_a.append(None)
                self.mid_b_b.append(None)

        # output proj
        self.w_out = nn.Parameter(torch.empty(self.n_concepts, self.dim, self.out_dim))
        self.b_out = nn.Parameter(torch.zeros(self.n_concepts, self.out_dim))

        # ----- init -----
        nn.init.kaiming_uniform_(self.w1_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w1_b, a=math.sqrt(5))
        for wa, wb in zip(self.mid_w_a, self.mid_w_b):
            nn.init.kaiming_uniform_(wa, a=math.sqrt(5))
            nn.init.kaiming_uniform_(wb, a=math.sqrt(5))
        nn.init.zeros_(self.w_out)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().cpu().numpy().item()
        alpha = self.dim if alpha is None or alpha == 0 else alpha
        self.scale = float(alpha) / float(self.dim)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        self.multiplier = multiplier
        self.org_module = org_module
        self.use_prompt_tuning = False


    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    # ---- helpers ----
    def _first_block(self, x):
        # x: (B,T,in_dim)  -> h: (B,T,N,dim)
        if self.prenorm:
            x = self.input_norm(x)
        if self.use_bias and (self.b1_a is not None):
            h_a = torch.einsum("btd,ndh->btnh", x, self.w1_a) + self.b1_a.view(1,1,self.n_concepts,self.dim)
            h_b = torch.einsum("btd,ndh->btnh", x, self.w1_b) + self.b1_b.view(1,1,self.n_concepts,self.dim)
        else:
            h_a = torch.einsum("btd,ndh->btnh", x, self.w1_a)
            h_b = torch.einsum("btd,ndh->btnh", x, self.w1_b)
        h = torch.sigmoid(h_a) * h_b
        h = self.dropout(h)
        return h

    def _mid_block(self, h, wi, wj, bi, bj, norm: nn.Module | None):
        # h: (B,T,N,dim) -> (B,T,N,dim)
        if self.prenorm and (norm is not None):
            h = norm(h)
        if self.use_bias and (bi is not None):
            h_a = torch.einsum("btnh,nhh->btnh", h, wi) + bi.view(1,1,self.n_concepts,self.dim)
            h_b = torch.einsum("btnh,nhh->btnh", h, wj) + bj.view(1,1,self.n_concepts,self.dim)
        else:
            h_a = torch.einsum("btnh,nhh->btnh", h, wi)
            h_b = torch.einsum("btnh,nhh->btnh", h, wj)
        h = torch.sigmoid(h_a) * h_b
        h = self.dropout(h)
        return h

    def _last_proj(self, h):
        # h: (B,T,N,dim) -> y: (B,T,out_dim)
        y_all = torch.einsum("btnh,nhd->btnd", h, self.w_out) + self.b_out.view(1,1,self.n_concepts,self.out_dim)
        y_all = self.resid_dropout(y_all)
        y = y_all.sum(dim=2)
        return y

    def _forward_all_concepts_mlp_glu(self, x):
        h = self._first_block(x)
        # mid blocks
        for k, (wa, wb, ba, bb) in enumerate(zip(self.mid_w_a, self.mid_w_b, self.mid_b_a, self.mid_b_b)):
            h = self._mid_block(h, wa, wb, ba, bb, self.mid_norms[k] if self.prenorm else None)
        y = self._last_proj(h)
        return y

    def forward(self, x):
        # x: (B, T, in_dim)
        if not self.use_prompt_tuning:
            return self.org_forward(x)
        up = self._forward_all_concepts_mlp_glu(x)
        return self.org_forward(x) + up * self.multiplier * self.scale



class EASLayer_MoE_Dense(nn.Module):
    def __init__(
        self,
        eas_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=128,                      # H (expert hidden)
        alpha=1,
        init_size=1, hidden_size=256, num_embeddings=77, task_id=1, attention_gate=None, args=None,  # compat
        glu_type: str = "glu",        # "glu" | "swi"
        num_experts: int = 4,         # E
        top_k: int = 1,               
        router_noise_std: float = 0.0,
        keeptok: float = 0.0,         # 0~1
        lb_coef: float = 1e-2,
        cov_coef: float = 1e-3,
        prenorm_router: bool = True,
        ffn_norm: str = "rmsnorm",
        dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        use_bias: bool | None = None,
        n_concepts=1,
        sparse_compute=True,
        depth=1,
    ):
        super().__init__()
        self.eas_name = eas_name
        self.multiplier = float(multiplier)
        self.dim = int(dim)                 # H
        self.org_module = org_module
        self.use_prompt_tuning = False

        self.in_dim  = org_module.in_features
        self.out_dim = org_module.out_features

        self.args = args or type("Args", (), {})()
        self.use_bias = bool(getattr(self.args, "use_bias", True if use_bias is None else use_bias))

        self.glu_type = glu_type.lower()
        assert self.glu_type in ("glu", "swi")
        self.num_experts = int(num_experts)  # E
        self.top_k = int(top_k); assert 1 <= self.top_k <= self.num_experts
        self.router_noise_std = float(router_noise_std)
        self.keeptok = float(keeptok)
        self.lb_coef = float(lb_coef)
        self.cov_coef = float(cov_coef)
        self.prenorm_router = bool(prenorm_router)

        # norms & drops
        self.norm_type = ffn_norm.lower()
        self.prenorm = bool(getattr(self.args, "ffn_prenorm", True))
        self.input_norm = make_shared_norm(self.norm_type, self.in_dim) if self.prenorm else nn.Identity()
        self.router_norm = make_shared_norm(self.norm_type, self.in_dim) if self.prenorm_router else nn.Identity()
        self.dropout = nn.Dropout(float(getattr(self.args, "ffn_dropout", dropout_p)))
        self.resid_dropout = nn.Dropout(float(getattr(self.args, "ffn_resid_dropout", resid_dropout_p)))

        # ---- Dense experts as concatenated matrices (no per-expert gather) ----
        E, Din, H, O = self.num_experts, self.in_dim, self.dim, self.out_dim

        # 1st layer fused weights: [W1a|W1b] over experts concatenated on last dim
        self.w1ab_cat = nn.Parameter(torch.empty(Din, E * 2 * H))        # (Din, E*2H)
        self.b1ab_cat = nn.Parameter(torch.zeros(E * 2 * H)) if self.use_bias else None

        # 2nd layer weights concatenated vertically: block rows per expert
        self.wout_cat = nn.Parameter(torch.empty(E * H, O))              # (E*H, O)
        self.bout_all = nn.Parameter(torch.zeros(E, O))                  # (E, O) (expert-wise bias)

        # keeptok dense FFN (single)
        self.keep_w1_a = nn.Parameter(torch.empty(Din, H))
        self.keep_w1_b = nn.Parameter(torch.empty(Din, H))
        self.keep_b1_a = nn.Parameter(torch.zeros(H)) if self.use_bias else None
        self.keep_b1_b = nn.Parameter(torch.zeros(H)) if self.use_bias else None
        self.keep_w_out = nn.Parameter(torch.empty(H, O))
        self.keep_b_out = nn.Parameter(torch.zeros(O))

        self.silu = nn.SiLU(inplace=True)

        # init
        for p in [self.w1ab_cat, self.keep_w1_a, self.keep_w1_b]:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        nn.init.zeros_(self.wout_cat)
        nn.init.zeros_(self.keep_w_out)
        # router
        self.router = nn.Linear(Din, E, bias=True)
        nn.init.zeros_(self.router.weight); nn.init.zeros_(self.router.bias)

        # scale (LoRA-style)
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().cpu().numpy().item()
        alpha = self.dim if alpha is None or alpha == 0 else alpha
        self.scale = float(alpha) / float(self.dim)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        self.last_aux_losses = {"moe/lb_loss": torch.tensor(0.0),
                                "moe/cov_loss": torch.tensor(0.0),
                                "moe/entropy":  torch.tensor(0.0)}

        self._init_identity()



    def _init_identity(self):
        nn.init.zeros_(self.wout_cat)     # (E*H, O)
        if self.bout_all is not None:
            nn.init.zeros_(self.bout_all) # (E, O)

        nn.init.zeros_(self.keep_w_out)   # (H, O)
        if self.keep_b_out is not None:
            nn.init.zeros_(self.keep_b_out)


    # hook
    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _apply_glu(self, a, b):
        return (self.silu(a) if self.glu_type == "swi" else torch.sigmoid(a)) * b

    def _route_dense_mask(self, x):
        """
        Dense gates (B,T,E); if top_k < E, mask others to 0 and renormalize.
        """
        z = self.router_norm(x)
        logits = self.router(z)  # (B,T,E)
        if self.router_noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise_std
        gates = F.softmax(logits, dim=-1)  # (B,T,E)

        if self.top_k < self.num_experts:
            topv, topi = torch.topk(gates, self.top_k, dim=-1)        # (B,T,K)
            mask = torch.zeros_like(gates)                             # (B,T,E)
            mask.scatter_(dim=-1, index=topi, src=torch.ones_like(topv))
            gates = gates * mask
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        return gates  # (B,T,E)

    def _dense_keep_ffn(self, x_flat):
        Ha = x_flat @ self.keep_w1_a
        Hb = x_flat @ self.keep_w1_b
        if self.use_bias and self.keep_b1_a is not None:
            Ha = Ha + self.keep_b1_a
            Hb = Hb + self.keep_b1_b
        H = self._apply_glu(Ha, Hb)
        H = self.dropout(H)
        y = H @ self.keep_w_out + self.keep_b_out
        y = self.resid_dropout(y)
        return y.contiguous()  # (BT, O)

    def _aux_losses(self, gates_full, H_act_unnorm):
        B, T, E = gates_full.shape
        BT = B * T
        H = H_act_unnorm.size(-1)
        eps = 1e-9

        # load-balance
        importance = gates_full.sum(dim=(0,1))                          # (E,)
        load = (gates_full > (1.0 / E)).float().sum(dim=(0,1))          # (E,)
        imp_mean = importance.mean(); load_mean = load.float().mean()
        lb_loss = (importance.var(unbiased=False)/(imp_mean.clamp_min(eps)**2 + eps)) + \
                  (load.float().var(unbiased=False)/(load_mean.clamp_min(eps)**2 + eps))
        lb_loss = self.lb_coef * lb_loss

        # covariance (gate-weighted expert mean of H)
        g_flat = gates_full.reshape(BT, E)                              # (BT,E)
        sum_h = (H_act_unnorm * g_flat.unsqueeze(-1)).sum(dim=0)        # (E,H)
        cnt   = g_flat.sum(dim=0, keepdim=True).t().clamp_min(eps)      # (E,1)
        mean_h = sum_h / cnt                                            # (E,H)

        mean_h = mean_h - mean_h.mean(dim=0, keepdim=True)
        cov = (mean_h @ mean_h.t()) / (H + eps)                         # (E,E)
        off = cov - torch.diag(torch.diag(cov))
        cov_loss = self.cov_coef * (off.pow(2).mean())

        entropy = -(gates_full.clamp_min(eps) * gates_full.clamp_min(eps).log()).sum(dim=-1).mean()

        self.last_aux_losses["moe/lb_loss"] = lb_loss.detach()
        self.last_aux_losses["moe/cov_loss"] = cov_loss.detach()
        self.last_aux_losses["moe/entropy"]  = entropy.detach()
        return lb_loss + cov_loss

    def forward(self, x):
        if not self.use_prompt_tuning:
            return self.org_forward(x)

        x_in = x
        x_adapt = self.input_norm(x) if self.prenorm else x

        B, T, Din = x_in.shape
        BT = B * T
        E, H, O = self.num_experts, self.dim, self.out_dim

        gates = self._route_dense_mask(x_in)         # (B,T,E)
        g_flat = gates.reshape(BT, E).contiguous()

        x_flat = x_adapt.reshape(BT, Din).contiguous()
        Hab_all = x_flat @ self.w1ab_cat             # (BT, E*2H)
        if self.b1ab_cat is not None:
            Hab_all = Hab_all + self.b1ab_cat
        Hab_all = Hab_all.view(BT, E, 2*H)
        Ha, Hb = Hab_all[..., :H], Hab_all[..., H:]
        H_act = self._apply_glu(Ha, Hb)              # (BT,E,H)

        aux = self._aux_losses(gates, H_act)

        H_drop = self.dropout(H_act)
        H_gated = H_drop * g_flat.unsqueeze(-1)      # (BT,E,H)

        H_cat = H_gated.reshape(BT, E*H)
        Y = H_cat @ self.wout_cat                    # (BT,O)
        Y = Y + g_flat @ self.bout_all               # (BT,O)
        Y = self.resid_dropout(Y)
        up = Y.view(B, T, O)

        if self.keeptok > 0.0:
            y_keep = self._dense_keep_ffn(x_flat).view(B, T, O)
            up = (1.0 - self.keeptok) * up + self.keeptok * y_keep

        self._last_aux_total = aux

        return self.org_forward(x_in) + up * self.multiplier * self.scale












