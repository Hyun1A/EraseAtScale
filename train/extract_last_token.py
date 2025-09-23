# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py
# - https://github.com/Con6924/SPM

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import sys
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


# -----------------------------------------------------------------------------
# Local imports
# -----------------------------------------------------------------------------
# Ensure project root on path (one level up from current script)
sys.path[0] = "/".join(sys.path[0].split("/")[:-1])

from src.models.merge_eas import * 

from src.engine.train_util import * 
from src.engine.dataset_per_concept import build_triplet_loader_from_tmms
from src.engine.trainer_gate import *

from src.configs import config as config_pkg 
from src.configs import prompt as prompt_pkg  
from src.configs.config import RootConfig 


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
# Training (single target run)
# -----------------------------------------------------------------------------

def train_single_target(
    config: RootConfig,
    args: argparse.Namespace,
    target, ## --> target
    tokenizer, 
    text_encoder, 
    unet, 
    noise_scheduler, 
    pipe,
    amp_dtype: Optional[torch.dtype],
) -> None:

    # 2) Loader
    replace_word = config.replace_word
    data_loader, (X_target, X_mapping, X_anchor) = build_triplet_loader_from_tmms(
        target=target,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        config=config,
        replace_word=replace_word,
        batch_size=None if args.batch_size==1 else args.batch_size,     
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        conf_bound=args.conf,
        mapping_type=args.mapping_type,
        n_top=args.n_top,
    )

    flush()
    print("Done.")


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
    if args.lr != -1:
        config.train.lr = args.lr

    config.train.skip_learned = args.skip_learned

    # Experiment naming
    # Uses the *first* prompt for guide replacement; safe since we reassign in loop below too
    exp_name = (
        config.save.path.split("/")[-2]
        .replace("guide#", f"guide{args.guidance_scale}")
        .replace("pal#", f"pal{config.train.pal}")
        .replace("gate_rank#", f"gr{config.network.init_size}")
    )
    exp_name += f"_d{args.depth}_c{args.conf}"
    exp_name += f"_r{config.network.rank}"
    exp_name += f"_anc_{args.anchor_type}_map_{args.mapping_type}_{args.n_top}"

    if "moe" in args.arch_type.lower():
        exp_name += f"_{args.net_type.lower()}"
        exp_name += f"_{args.arch_type.lower()}_{args.glu_type}_E{args.n_experts}_k{args.top_k}_keep{args.keeptok}"
        exp_name += f"_b{args.dataset_n_batch}_lr{config.train.lr}_it{args.iterations}"

    config.save.path = "/".join(config.save.path.split("/")[:-2] + [exp_name] + [config.save.path.split("/")[-1]])




def main(args: argparse.Namespace) -> None:
    config = config_pkg.load_config_from_yaml(args.config_file)

    if config.replace_word != 'explicit_concept':
        prompts = prompt_pkg.load_prompts(config.prompts_file)
    else:
        prompts = ["explicit_concept"]

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

    for idx, p in enumerate(prompts):
        # prompt window selection
        if (idx < config.train.st_prompt_idx) or (idx > config.train.end_prompt_idx):
            continue

        # Logging & save path per-target
        config.logging.prompts = [s.replace("[target]", p) if "[target]" in s else s for s in base_logging_prompts]
        config.save.path = base_path.replace(config.replace_word.upper(), p.replace(" ", "_"))

        os.makedirs(config.save.path, exist_ok=True)
        last_ckpt = Path(config.save.path) / f"{config.save.name}_last.safetensors"
        if config.train.skip_learned and last_ckpt.is_file():
            print(f"{idx} {p} has already been trained")
            continue

        print("\n" + "-" * 80)
        print(f"[{idx}] target={p} | pal={config.train.pal}")
        print("-" * 80 + "\n")

        seed_everything(config.train.train_seed)
        train_single_target(config, args, p, tokenizer, text_encoder, unet, noise_scheduler, pipe, amp_dtype=amp_dtype)


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
    parser.add_argument("--lr", type=float, default=-1) 
    parser.add_argument("--conf", type=float, default=0.9)
    parser.add_argument("--anchor_type", type=str, default="mapping")
    parser.add_argument("--mapping_type", type=str, default="similar") 
    parser.add_argument("--n_top", type=int, default=3) 
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="auto",  # auto|bf16|fp16|off
        help="Mixed precision dtype: auto(bf16 if supported, else fp16), bf16, fp16, off",
    )

    args = parser.parse_args()
    main(args)


