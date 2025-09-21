# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py
# - https://github.com/Con6924/SPM
from __future__ import annotations

import os, math, gc, random
from typing import List, Dict, Tuple, Iterable, Optional
import torch
from torch.utils.data import Dataset, DataLoader


import argparse
import gc
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd
from pathlib import Path


import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from torch.cuda.amp import autocast


from src.engine.pm_util.utils.tmm import TMMTorch


from src.engine.pm_util.utils.tmm import TMMTorch
from src.engine.pm_util.utils.pca import pca_reconstruct, pca_reduce
from src.engine.pm_util.utils.parsing import stack_embeds, stack_embeds_last_token, stack_embeds_eos, stack_embeds_none, encode_prompt
from src.engine.pm_util.utils.sampling import sample_in_conf_interval
from src.engine.pm_util.utils.merge_tmm import merge_tmm_models
from src.engine.pm_util.utils.transport import optimal_transport


from src.configs import config as config_pkg
from src.configs import prompt as prompt_pkg
from src.engine import train_util



class TripletTensorDataset(Dataset):
    """Yields (targets, mapping, anchors) where
    - targets: (2, D/2)
    - mapping: (2, D/2)
    - anchors: (n_anc, 2, D/2)
    This matches the trainer's expected shapes for `torch.cat([targets, mapping, uncond, anchors], dim=0)`.
    """
    def __init__(self, X_target: torch.Tensor, X_mapping: torch.Tensor, X_anchor: torch.Tensor, n_anc: int = 32):
        assert X_target.ndim == 3 and X_mapping.ndim == 3 and X_anchor.ndim == 3, "Pre-reshape to (B, 2, D/2) required"
        self.X_target = X_target
        self.X_mapping = X_mapping
        self.X_anchor = X_anchor
        self.n_anc = int(n_anc)

    def __len__(self) -> int:
        return self.X_target.size(0)

    def __getitem__(self, idx: int):
        # pick anchors independently each sample
        anc_idx = torch.randint(low=0, high=self.X_anchor.size(0), size=(self.n_anc,))
        targets = self.X_target[idx]          # (2, D/2)
        mapping = self.X_mapping[idx]         # (2, D/2)
        anchors = self.X_anchor[anc_idx]      # (n_anc, 2, D/2)

        return targets, mapping, anchors
    



# -----------------------------------------------------------------------------
# Prompt / embedding caches
# -----------------------------------------------------------------------------

def build_prompt_scripts_list(config) -> Tuple[List[str], str]:
    df = pd.read_csv(config.scripts_file)
    return df["prompt"].tolist(), config.replace_word


def encode_list(tokenizer, text_encoder, texts: List[str]) -> torch.Tensor:
    return train_util.encode_prompts(tokenizer, text_encoder, texts)



# -----------------------------------------------------------------------------
# Similar-word anchors (explicit/actor/artist)
# -----------------------------------------------------------------------------

def _read_anchor_csv(path: str, pick_col: Optional[int] = None) -> List[str]:
    df = pd.read_csv(path)
    rows = list(df.itertuples(index=False, name=None))
    if pick_col is None:
        # expect 3 columns like (idx, phrase, type) -> return phrase
        return [a for _, a, _ in rows]
    return [row[pick_col] for row in rows]


def collect_words(
    tokenizer,
    text_encoder,
    replace_word: str,
    targets: List[str],
    mode="similar",
    n_top=5,
    emb_anchor=None
) -> List[str]:

    simWords_erase = [t for t in targets]

    if replace_word in ["explicit", "concept", "explicit_concept"]:
        anchor_pool = _read_anchor_csv("./configs/train_explicit/prompt_mapping.csv", pick_col=0)
    elif replace_word == "actor":
        anchor_pool = _read_anchor_csv("./configs/train_celeb/prompt_mapping.csv", pick_col=0)
    elif replace_word == "character":
        anchor_pool = _read_anchor_csv("./configs/train_char/prompt_mapping.csv", pick_col=0)
    else:
        anchor_pool = _read_anchor_csv("./configs/train_artist/prompt_mapping.csv", pick_col=0)
    

    # remove erased term from pool
    lowered = [a.lower() for a in anchor_pool]
    for s in simWords_erase:
        lowered = [it for it in lowered if s not in it]
    filtered_pool = lowered

    # embed erase and anchors, pick top-5 closest per first erase target
    emb_erase = encode_list(tokenizer, text_encoder, simWords_erase)
    # pack anchors in batches (100)
    B = 100
    if emb_anchor is None:
        emb_anchor_batches: List[torch.Tensor] = []
        for b in range(0, len(filtered_pool), B):
            emb_anchor_batches.append(encode_list(tokenizer, text_encoder, filtered_pool[b : b + B]))
        emb_anchor = torch.cat(emb_anchor_batches, dim=0) if emb_anchor_batches else torch.empty(0)

    e = emb_erase.view(len(simWords_erase), -1)
    a = emb_anchor.view(len(filtered_pool), -1)
    e = e / e.norm(2, dim=1, keepdim=True)
    a = a / a.norm(2, dim=1, keepdim=True)
    sim = e @ a.T
    _vals, idx = sim.sort()
    if idx.numel() == 0:
        return []
    
    if mode == "similar":
        top = [filtered_pool[i] for i in idx[0][-n_top:].tolist()]
    elif mode == "dissimilar":
        top = [filtered_pool[i] for i in idx[0][:n_top].tolist()]
    elif mode == "include_mid":
        top = [filtered_pool[i] for i in idx[0][-(n_top-1):].tolist()]
        top += [filtered_pool[i] for i in idx[0][len(idx[0])//2:len(idx[0])//2+1].tolist()]
        # breakpoint()

    # sorted_list = [filtered_pool[i] for i in idx[0].tolist()]    

    print(f"selected: {top}")

    return top




# -----------------------------------------------------------------------------
# Token-embed caches & TMMs
# -----------------------------------------------------------------------------

def _build_token_caches(target, tgt_prompts, tokenizer, text_encoder, sim_words, sim_w_prompts, replace_word="concept", cache_dir: str = "./emb_cache"): 

    if replace_word != "explicit_concept":
        _full, embeds, _ = stack_embeds_last_token(tgt_prompts, text_encoder, tokenizer, " " + target)
    else:
        _full, embeds, _ = stack_embeds_eos(tgt_prompts, text_encoder, tokenizer,)

    embeds_sim_dict = {}
    for w, w_prompts in zip(sim_words, sim_w_prompts):
        if replace_word != "explicit_concept":
            _f, w_embeds, _s = stack_embeds_last_token(w_prompts, text_encoder, tokenizer, " " + w)
        else:
            _f, w_embeds, _s = stack_embeds_eos(w_prompts, text_encoder, tokenizer,)
        
        embeds_sim_dict[w] = w_embeds

    return embeds, embeds_sim_dict








def _build_tmm_models(target, embeds: torch.Tensor, embeds_sim_dict, reduce=True, k=32, replace_word="concept", cache_dir: str = "./pm_cache"):
    cache_dir = f"{cache_dir}/{replace_word}"
    os.makedirs(cache_dir, exist_ok=True)
    tgt_fn = Path(cache_dir) / f"target_{target}.pt"
    if tgt_fn.is_file():
        tmm_target = torch.load(tgt_fn)
    else:
        with autocast(enabled=False):
            if reduce:
                embeds, W, mean, var = pca_reduce(embeds, k=32)
                print(f"Explainable variance: {var}")
            dof_init = 2
            

            tmm_target = TMMTorch(
                n_components=4,
                covariance_type="diag",
                tied_covariance=False,
                shrinkage_alpha=0.0,
                scale_floor_scale=0.0,
                min_cluster_weight=0.0,
                learn_dof=False,
                dof_init=dof_init,
                verbose=True,
                seed=42,
            ).fit(embeds)

        if reduce:
            tmm_target.basis = W
            tmm_target.mean = mean
                
        torch.save(tmm_target, tgt_fn)

    sim_fn = Path(cache_dir) / f"similar_{target}.pt"
    if sim_fn.is_file():
        tmm_mapping = torch.load(sim_fn)
    else:
        tmm_mapping = {}
        for w, e in embeds_sim_dict.items():

            if reduce:
                e, W, mean, var = pca_reduce(e, k=32)
                print(f"Explainable variance: {var}")
            dof_init = 2

            tmm_mapping[w] = TMMTorch(
                n_components=4,
                covariance_type="diag",
                tied_covariance=False,
                shrinkage_alpha=0.0,
                scale_floor_scale=0.0,
                min_cluster_weight=0.0,
                learn_dof=False,
                dof_init=dof_init,
                verbose=True,
                seed=42,
            ).fit(e)

            if reduce:
                tmm_mapping[w].basis = W
                tmm_mapping[w].mean = mean

        torch.save(tmm_mapping, sim_fn)
    return tmm_target, tmm_mapping


def build_triplet_loader_from_tmms(
    *,
    target,
    tokenizer,
    text_encoder,
    config,
    replace_word,
    batch_size=None,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    reduce=True,
    k=32,
    conf_bound=0.9,
    mapping_type="similar",
    n_top=3,
):
    """Given config + sim_words, create (DataLoader, meta) for training.
    Returns: data_loader, (X_target, X_mapping, X_anchor)
    """

    n_top=3

    total_token_num = 1 # sum(tokenizer(target).attention_mask)-2


    if replace_word == "explicit_concept":
        # templates_path = getattr(config.train, "templates_path", "./configs/train_explicit/templates.txt")
        # tgt_prompts = getattr(config.train, "templates_path", "./configs/train_explicit/templates.txt")
        map_words = ["safe_concept"]
        tgt_prompts = pd.read_csv("./configs/train_explicit/prompt_target.csv")['explicit_concept'].tolist()
        sim_w_prompts = [pd.read_csv("./configs/train_explicit/prompt_mapping.csv")['explicit_concept'].tolist()]
        
    elif replace_word == "character":
        map_words = collect_words(tokenizer, text_encoder, replace_word, [target], mode=mapping_type, n_top=n_top)
        templates_path = getattr(config.train, "templates_path", "./configs/train_char/templates.txt")
    elif replace_word == "style":
        map_words = collect_words(tokenizer, text_encoder, replace_word, [target], mode=mapping_type, n_top=n_top)
        templates_path = getattr(config.train, "templates_path", "./configs/train_artist/templates.txt")
    else:
        map_words = collect_words(tokenizer, text_encoder, replace_word, [target], mode=mapping_type, n_top=n_top)
        templates_path = getattr(config.train, "templates_path", "./configs/train_celeb/templates.txt")

    if replace_word != "explicit_concept":
        with open(templates_path, "r", encoding="utf-8") as f:
            templates = f.readlines()
        tgt_prompts = [t.rstrip("").format(" " + target) for t in templates]
        sim_w_prompts = []
        for w in map_words:
            sim_w_prompts.append([t.rstrip("").format(" " + w) for t in templates])

    embeds, embeds_sim = _build_token_caches(target, tgt_prompts, tokenizer, text_encoder, map_words, sim_w_prompts, replace_word)

    cache_dir = f"./pm_cache/{mapping_type}_{n_top}"
    tmm_target, tmm_mapping = _build_tmm_models(target, embeds, embeds_sim, reduce, k, replace_word, cache_dir)


    # sample within confidence regions
    n_rand, conf_low, conf_high = 2*1024, 1e-8, conf_bound

    X_target_redueced, _ = sample_in_conf_interval(
        tmm_target,
        n=n_rand*total_token_num,
        conf_low=conf_low,
        conf_high=conf_high,
        n_ref=800_000,
        helper_batch=65536,
        max_batches=300,
        pad_ll=0.0,
    )
    b, d = X_target_redueced.size()
    b = b // total_token_num


    if reduce:
        X_target_recon = pca_reconstruct(X_target_redueced, tmm_target.basis, tmm_target.mean)
    X_target = X_target_recon.detach().clone()
    X_target = X_target.reshape(b, total_token_num, -1)


    tmm_merged = merge_tmm_models(list(tmm_mapping.values()))

    X_mapping, _ = optimal_transport(tmm_target, tmm_merged, X_target_redueced, return_plan=True)
    if reduce:
        X_mapping_recon = pca_reconstruct(X_mapping, tmm_merged.basis, tmm_merged.mean)
    X_mapping = X_mapping_recon.detach().clone()
    X_mapping = X_mapping.reshape(b, total_token_num, -1)

    X_anchor = X_mapping

    ###### save concept dataset ######
    concept_data_dict = {"concept_tar": target, "concept_anc": map_words, \
                        "target": X_target, "mapping": X_mapping, "anchor": X_anchor}

    os.makedirs(f"./dataset/train_pairs/{mapping_type}_{n_top}/conf{conf_high}/{replace_word}/", exist_ok=True)
    train_data_path = f"./dataset/train_pairs/{mapping_type}_{n_top}/conf{conf_high}/{replace_word}/{target}.pt"
    torch.save(concept_data_dict, train_data_path)


    n_anc = getattr(config.train, "n_anc", 4)
    dataset = TripletTensorDataset(X_target, X_mapping, X_anchor, n_anc=n_anc)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
    return loader, (X_target, X_mapping, X_anchor)




def get_anchor_pool(
    tokenizer,
    text_encoder,
    replace_word: str,
    targets: List[str],
    emb_anchor = None,
) -> List[str]:

    simWords_erase = [t for t in targets]

    if replace_word in ["explicit", "concept", "explicit_concept"]:
        anchor_pool = _read_anchor_csv("./configs/train_explicit/prompt_mapping.csv", pick_col=0)
    elif replace_word == "actor":
        anchor_pool = _read_anchor_csv("./configs/train_celeb/prompt_mapping.csv", pick_col=0)
    elif replace_word == "character":
        anchor_pool = _read_anchor_csv("./configs/train_char/prompt_mapping.csv", pick_col=0)
    else:
        anchor_pool = _read_anchor_csv("./configs/train_artist/prompt_mapping.csv", pick_col=0)

    # remove erased term from pool
    lowered = [a.lower() for a in anchor_pool]
    for s in simWords_erase:
        lowered = [it for it in lowered if s not in it]
    filtered_pool = lowered

    # pack anchors in batches (100)
    if emb_anchor is None:
        B = 100
        emb_anchor_batches: List[torch.Tensor] = []
        for b in range(0, len(filtered_pool), B):
            emb_anchor_batches.append(encode_list(tokenizer, text_encoder, filtered_pool[b : b + B]))
        emb_anchor = torch.cat(emb_anchor_batches, dim=0) if emb_anchor_batches else torch.empty(0)

    return emb_anchor





def extract_text_embeddings(
    *,
    target_list,
    tokenizer,
    text_encoder,
    config,
    replace_word,
    batch_size=None,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    reduce=True,
    k=32,
    conf_bound=0.9,
):
    """Given config + sim_words, create (DataLoader, meta) for training.
    Returns: data_loader, (X_target, X_mapping, X_anchor)
    """

    sampling_type="base_w_char_in_word_w_mid_samp_top3"
    mode="similar"
    n_top=1


    emb_anchor = get_anchor_pool(tokenizer, text_encoder, replace_word, target_list)


    full_embeds_target = []
    full_embeds_anchor = []

    tgt_prompts = []
    sim_w_prompts = []

    for target in target_list:
        if replace_word == "explicit_concept":
            # templates_path = getattr(config.train, "templates_path", "./configs/train_explicit/templates.txt")
            # tgt_prompts = getattr(config.train, "templates_path", "./configs/train_explicit/templates.txt")
            map_words = ["safe_concept"]
            tgt_prompts = pd.read_csv("./configs/train_explicit/prompt_target.csv")['explicit_concept'].tolist()
            sim_w_prompts = [pd.read_csv("./configs/train_explicit/prompt_mapping.csv")['explicit_concept'].tolist()]
        
        elif replace_word == "character":
            map_words = collect_words(tokenizer, text_encoder, replace_word, [target], mode=mode, n_top=n_top, emb_anchor=emb_anchor)
        elif replace_word == "style":
            map_words = collect_words(tokenizer, text_encoder, replace_word, [target], mode=mode, n_top=n_top, emb_anchor=emb_anchor)
        else:
            map_words = collect_words(tokenizer, text_encoder, replace_word, [target], mode=mode, n_top=n_top, emb_anchor=emb_anchor)

        if replace_word != "explicit_concept":
            templates = ["An image of {}"]

            tgt_prompts.append([t.rstrip("").format(" " + target) for t in templates][0])
            sim_w_prompts.append([t.rstrip("").format(" " + map_words[0]) for t in templates][0])


    full_embeds_target = encode_prompt(tgt_prompts, "cuda", text_encoder, tokenizer)
    full_embeds_anchor = encode_prompt(sim_w_prompts, "cuda", text_encoder, tokenizer)




    return full_embeds_target, full_embeds_anchor, tgt_prompts, sim_w_prompts

