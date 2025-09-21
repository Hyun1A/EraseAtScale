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
from src.engine.pm_util.utils.parsing_eos import stack_embeds
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
    n_top=5
) -> List[str]:

    simWords_erase = [t for t in targets]

    if replace_word in ["explicit", "concept", "explicit_concept"]:
        df = pd.read_csv("./configs/train_explicit/prompt_mapping.csv")
        # keep first column (prompt) from tuples
        anchor_pool = list(set([row[-1] for row in df.itertuples(index=False, name=None)]))

    # actor / artist
    elif replace_word == "actor":
        anchor_pool = _read_anchor_csv("./configs/train_celeb/prompt_mapping.csv", pick_col=0)
    else:
        df = pd.read_csv("./anchors/mass_surr_words_1734artists.csv")
        anchor_pool = list(set([row[-1] for row in df.itertuples(index=False, name=None)]))
    

    # remove erased term from pool
    lowered = [a.lower() for a in anchor_pool]
    for s in simWords_erase:
        lowered = [it for it in lowered if s not in it]
    filtered_pool = lowered

    # embed erase and anchors, pick top-5 closest per first erase target
    emb_erase = encode_list(tokenizer, text_encoder, simWords_erase)
    # pack anchors in batches (100)
    B = 100
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

    return top




# -----------------------------------------------------------------------------
# Token-embed caches & TMMs
# -----------------------------------------------------------------------------

def _build_token_caches(target, templates_path: str, tokenizer, text_encoder, sim_words, cache_dir: str = "./emb_cache_eos"): 
    os.makedirs(cache_dir, exist_ok=True)
    with open(templates_path, "r", encoding="utf-8") as f:
        templates = f.readlines()
    
    tgt_prompts = [t.rstrip("").format(" " + target) for t in templates]

    tgt_path = Path(cache_dir) / f"target_{target}.pt"
    if tgt_path.is_file():
        embeds = torch.load(tgt_path)
    else:
        _full, embeds, _ = stack_embeds(tgt_prompts, text_encoder, tokenizer, " " + target)
        torch.save(embeds, tgt_path)

    sim_path = Path(cache_dir) / f"similar_{target}.pt"
    if sim_path.is_file():
        embeds_map_dict = torch.load(sim_path)
    else:
        embeds_map_dict = {}
        for w in sim_words:
            w_prompts = [t.rstrip("").format(" " + w) for t in templates]
            _f, w_embeds, _s = stack_embeds(w_prompts, text_encoder, tokenizer, " " + w)
            embeds_map_dict[w] = w_embeds
        torch.save(embeds_map_dict, sim_path)

    return embeds, embeds_map_dict








def _build_tmm_models(target, embeds: torch.Tensor, embeds_map_dict, reduce=True, k=32, cache_dir: str = "./pm_cache_eos"):

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
        for w, e in embeds_map_dict.items():

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
):
    """Given config + sim_words, create (DataLoader, meta) for training.
    Returns: data_loader, (X_target, X_mapping, X_anchor)
    """

    total_token_num = 1

    if replace_word == "explicit_concept":
        map_words = collect_words(tokenizer, text_encoder, replace_word, [target], mode="dissimilar", n_top=1)
    else:
        map_words = collect_words(tokenizer, text_encoder, replace_word, [target])


    templates_path = getattr(config.train, "templates_path", "./configs/train_celeb/templates.txt")

    embeds, embeds_map = _build_token_caches(target, templates_path, tokenizer, text_encoder, map_words)
    tmm_target, tmm_mapping = _build_tmm_models(target, embeds, embeds_map, reduce, k)

    # breakpoint()

    # sample within confidence regions
    n_rand, conf_low, conf_high = 1024*1024, 1e-8, 0.99999

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



    # reshape to (B, 2, D/2)
    X_anchor = X_mapping



    ###### save concept dataset ######
    concept_data_dict = {"concept_tar": target, "concept_anc": map_words, \
                        "target": X_target, "mapping": X_mapping, "anchor": X_anchor}

    os.makedirs(f"./dataset/train_pairs_eos_conf{conf_high}/{replace_word}/", exist_ok=True)
    train_data_path = f"./dataset/train_pairs_eos_conf{conf_high}/{replace_word}/{target}.pt"
    torch.save(concept_data_dict, train_data_path)



    n_anc = getattr(config.train, "n_anc", 4)
    dataset = TripletTensorDataset(X_target, X_mapping, X_anchor, n_anc=n_anc)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
    return loader, (X_target, X_mapping, X_anchor)

