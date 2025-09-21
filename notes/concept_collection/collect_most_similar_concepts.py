
# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py
# - https://github.com/Con6924/SPM

from __future__ import annotations

import argparse
import gc
import math
import os, sys
sys.path[0] = "/".join(sys.path[0].split("/")[:-1])

import random
import sys
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from safetensors.torch import load_file, save_file
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import wandb



from src.engine.train_util import * 
from src.engine.dataset_per_concept import build_triplet_loader_from_tmms
from src.engine.trainer_gate import *

from src.configs import config as config_pkg 
from src.configs import prompt as prompt_pkg  
from src.configs.config import RootConfig 

from src.configs import config as config_pkg
from src.configs import prompt as prompt_pkg
from src.engine import train_util




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
) -> List[str]:

    simWords_erase = [t for t in targets]

    if replace_word in ["explicit", "concept", "explicit_concept"]:
        anchor_pool = _read_anchor_csv("./configs/train_explicit/prompt_preserve.csv", pick_col=0)
    elif replace_word == "actor":
        anchor_pool = _read_anchor_csv("./configs/train_celeb/prompt_preserve.csv", pick_col=0)
    elif replace_word == "character":
        anchor_pool = _read_anchor_csv("./configs/train_char/prompt_preserve.csv", pick_col=0)
    else:
        anchor_pool = _read_anchor_csv("./configs/train_artist/prompt_preserve.csv", pick_col=0)
    

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
# Model assembly helpers
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Noise helpers
# -----------------------------------------------------------------------------

def load_or_make_noise(
    unet_modules: Dict[str, torch.nn.Module],
    save_path: Path,
    rand_scale: float,
) -> Dict[str, torch.Tensor]:
    noise_path = save_path.parent / f"{save_path.name}_noise.safetensors"
    if noise_path.is_file():
        print("loading cached noise ...")
        noise = load_file(str(noise_path))
        return {k: v.to(DEVICE_CUDA) for k, v in noise.items()}

    print("generating noise ...")
    rank, in_dim = 1, 768
    w_down = torch.randn(rank, in_dim, device=DEVICE_CUDA)
    noise_dict: Dict[str, torch.Tensor] = {}
    for key, mod in unet_modules.items():
        W = mod.weight.data
        out_dim, in_dim = W.size()
        noise = rand_scale * w_down / in_dim
        noise_dict[key] = W.norm() * noise
    save_file(noise_dict, str(noise_path), None)
    return noise_dict


# -----------------------------------------------------------------------------
# Top-level runner
# -----------------------------------------------------------------------------

def update_config_from_args(config: RootConfig, args: argparse.Namespace) -> None:
    if args.st_prompt_idx != -1:
        config.train.st_prompt_idx = args.st_prompt_idx
    if args.end_prompt_idx != -1:
        config.train.end_prompt_idx = args.end_prompt_idx
    if args.gate_rank != -1:
        config.network.init_size = args.gate_rank
        config.network.hidden_size = args.gate_rank
        config.network.continual_rank = args.gate_rank

    if args.pal != -1:
        config.train.pal = args.pal
    if args.resume_stage != -1:
        config.train.resume_stage = args.resume_stage
    if args.lora_rank != -1:
        config.network.rank = args.lora_rank
    config.train.skip_learned = args.skip_learned

    # Experiment naming
    # Uses the *first* prompt for guide replacement; safe since we reassign in loop below too
    exp_name = (
        config.save.path.split("/")[-2]
        .replace("guide#", f"guide{args.guidance_scale}")
        .replace("pal#", f"pal{config.train.pal}")
        .replace("gate_rank#", f"gate_rank{config.network.init_size}")
    )

    exp_name += f"_last_token_conf{args.conf}_with_luigi_expand_uncond_with_bias"

    if args.lora_rank != -1:
        exp_name += f"_lora_rank{config.network.rank}"

    config.save.path = "/".join(config.save.path.split("/")[:-2] + [exp_name] + [config.save.path.split("/")[-1]])




def main(args: argparse.Namespace) -> None:
    config = config_pkg.load_config_from_yaml(args.config_file)
    prompts = prompt_pkg.load_prompts(config.prompts_file)

    update_config_from_args(config, args)

    base_path = config.save.path
    base_logging_prompts = list(config.logging.prompts)

    # AMP policy
    amp_dtype = select_amp_dtype(getattr(args, "amp_dtype", "auto"))

    # Components
    weight_dtype = config_pkg.parse_precision(config.train.precision)
    tokenizer, text_encoder, unet, noise_scheduler, pipe = load_sd_components(config)
    enable_eval_fwd_only(unet, text_encoder, weight_dtype)
    pipe.safety_checker = None
    set_matmul_precision()



    replace_word = config.replace_word
    if replace_word in ["explicit", "concept", "explicit_concept"]:
        anchor_pool = _read_anchor_csv("./configs/train_explicit/prompt_preserve.csv", pick_col=0)
    elif replace_word == "actor":
        anchor_pool = _read_anchor_csv("./configs/train_celeb/prompt_preserve.csv", pick_col=0)
    elif replace_word == "character":
        anchor_pool = _read_anchor_csv("./configs/train_char/prompt_preserve.csv", pick_col=0)
    else:
        anchor_pool = _read_anchor_csv("./configs/train_artist/prompt_preserve.csv", pick_col=0)


    mapping_selected = []
    for idx, p in enumerate(prompts):
        target = p
        map_words = collect_words(tokenizer, text_encoder, replace_word, [target])

        print(f"{idx}  target: {target}, selected: {map_words}")

        mapping_selected.append(map_words[-1])

        # if idx == 10:
        #     break


    mapping_unique = set(mapping_selected)
    mapping_selected = list(mapping_unique)
    anchor_unique = set(anchor_pool)
    anchor_diff = list(anchor_unique.difference(mapping_unique))
    num_comp = len(prompts) - len(anchor_unique)


    idx_diff = list(range(len(anchor_diff)))
    random.shuffle(idx_diff)
    idx_sel = idx_diff[:num_comp]

    mapping_comp = [anchor_diff[i] for i in idx_sel]
    # mapping_comp = []
    # for i in idx_sel: mapping_comp.append(anchor_diff[i])


    mapping_final = mapping_selected + mapping_comp

    remain_final = list(anchor_unique.difference(set(mapping_final)))

    mapping_dir = "/".join(args.config_file.split("/")[:-1])+"/prompt_mapping_new.csv"
    with open(mapping_dir, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([replace_word])
        for x in mapping_final:
            writer.writerow([x])

    remain_dir = "/".join(args.config_file.split("/")[:-1])+"/prompt_remain_new.csv"
    with open(remain_dir, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([replace_word])
        for x in remain_final:
            writer.writerow([x])




# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Config file for training.")
    parser.add_argument("--st_prompt_idx", type=int, default=-1)
    parser.add_argument("--end_prompt_idx", type=int, default=-1)
    parser.add_argument("--gate_rank", type=int, default=-1)
    parser.add_argument("--guidance_scale", type=float, default=-1)
    parser.add_argument("--pal", type=float, default=-1)
    parser.add_argument("--noise", type=str, default="")
    parser.add_argument("--resume_stage", type=int, default=-1)
    parser.add_argument("--lora_rank", type=int, default=-1)
    parser.add_argument("--mixup", type=bool, default=True)
    parser.add_argument("--skip_learned", type=bool, default=False)
    parser.add_argument("--rand", type=float, default=0.01)
    parser.add_argument("--arch_type", type=str, default="gate")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.9)
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="auto",  # auto|bf16|fp16|off
        help="Mixed precision dtype: auto(bf16 if supported, else fp16), bf16, fp16, off",
    )

    args = parser.parse_args()
    main(args)
