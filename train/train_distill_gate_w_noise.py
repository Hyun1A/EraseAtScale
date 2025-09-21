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
from src.models.eas_layers import(
    EASLayer_Gate,
    EASLayer_MLP,
    EASLayer_MLP_SwiGLU,
    EASLayer_Linear,
    EASLayer_FFN,
    EASLayer_FFN_GLU,
    EASLayer_MoE_Dense,
)

from src.models.eas_networks import(
    EASNetwork,
)



from src.engine.train_util import * 
from src.engine.sharded_dataset import build_global_shuffled_shards, GlobalShuffledShardDataset
from src.engine.trainer_distill import *

from src.configs import config as config_pkg 
from src.configs import prompt as prompt_pkg  
from src.configs.config import RootConfig 



# -----------------------------------------------------------------------------
# Model assembly helpers
# -----------------------------------------------------------------------------

def pick_arch(name: str):
    name = (name or "gate").lower()
    if name == "gate":
        return EASLayer_Gate
    elif args.arch_type == "mlp":
        return EASLayer_MLP
    elif args.arch_type == "mlp_swiglu":
        return EASLayer_MLP_SwiGLU
    elif args.arch_type == "linear":
        return EASLayer_Linear
    elif args.arch_type == "ffn":
        return EASLayer_FFN
    raise ValueError(f"Unsupported arch_type: {name}")


def build_network(
    unet,
    text_encoder,
    config: RootConfig,
    args: argparse.Namespace,
    weight_dtype: torch.dtype,
) -> EASNetwork:
    task_id = len(config.pretrained_model.safetensor)
    arch = pick_arch(args.arch_type)
    net = EASNetwork(
        unet,
        text_encoder,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        module=arch,
        continual=True,
        task_id=task_id,
        continual_rank=config.network.continual_rank,
        hidden_size=config.network.hidden_size,
        init_size=config.network.init_size,
        args=args,
    ).to(DEVICE_CUDA, dtype=weight_dtype)
    return net



def build_network_teacher(
    unet,
    text_encoder,
    config: RootConfig,
    args: argparse.Namespace,
    weight_dtype: torch.dtype,
    arch_type = "gate"
) -> EASNetwork:
    task_id = len(config.pretrained_model.safetensor)
    
    arch = pick_arch(arch_type)
    

    concepts_ckpt = []
    for domain_path in args.model_domains:
        concept_path = f"{domain_path}/{args.model_path[0]}"

        concepts_folder = os.listdir(concept_path)    
        
        for folder in concepts_folder:
            if os.path.isfile(os.path.join(concept_path, folder)):
                continue
                            
            for ckpt in os.listdir(os.path.join(concept_path,folder)):
                if ("last.safetensors" in ckpt):
                    concepts_ckpt.append(os.path.join(concept_path,folder,ckpt))

    model_paths = [Path(lp) for lp in concepts_ckpt]

    eas_modules, metadatas = zip(*[
        load_state_dict(model_path, weight_dtype) for model_path in model_paths
    ])

    # check if EASs are compatible
    assert all([metadata["rank"] == metadatas[0]["rank"] for metadata in metadatas])


    # get the erased concept
    net = EASNetwork(
        unet,
        text_encoder,
        rank=int(float(metadatas[0]["rank"])),
        multiplier=1.0,
        alpha=float(metadatas[0]["alpha"]),
        module=arch,
        continual=True,
        task_id=task_id,
        continual_rank=config.network.continual_rank,
        hidden_size=config.network.hidden_size,
        init_size=config.network.init_size,
        n_concepts=len(model_paths),
        args=args,
    ).to(DEVICE_CUDA, dtype=weight_dtype)  

    for k,v in net.named_parameters(): 
        print(f"{k:100}", v.shape, eas_modules[0][k].shape)
        for idx in range(len(eas_modules)):
            # try:
            if len(v.shape) > 1:
                v.data[idx,:] = eas_modules[idx][k]
            elif args.arch_type=="gate":
                v.data[idx] = eas_modules[idx][k]
            else:
                v.data = eas_modules[idx][k]
                    
        net.to(DEVICE_CUDA, dtype=weight_dtype)  



    return net



# -----------------------------------------------------------------------------
# Noise helpers
# -----------------------------------------------------------------------------

def load_or_make_noise(
    noise_type,
    unet_modules: Dict[str, torch.nn.Module],
    save_path: Path,
    rand_scale: float,
) -> Dict[str, torch.Tensor]:
    if noise_type == "row_wise":
        noise_path = save_path.parent / f"noise_{noise_type}.safetensors"
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


    if noise_type == "low_rank_1":
        noise_path = save_path.parent / f"noise_{noise_type}.safetensors"
        if noise_path.is_file():
            print("loading cached noise ...")
            noise = load_file(str(noise_path))
            return {k: v.to(DEVICE_CUDA) for k, v in noise.items()}

        print("generating noise ...")
        rank, in_dim = int(noise_type.split("_")[-1]), 768
        w_down = torch.randn(rank, in_dim, device=DEVICE_CUDA)
        noise_dict: Dict[str, torch.Tensor] = {}
        for key, mod in unet_modules.items():
            W = mod.weight.data
            out_dim, in_dim = W.size()
            w_up = torch.randn(out_dim, rank, device=DEVICE_CUDA)            
            noise = w_up@w_down
            noise = rand_scale * noise / (in_dim*out_dim)**0.5
            noise_dict[key] = W.norm() * noise
        save_file(noise_dict, str(noise_path), None)



    return noise_dict



# -----------------------------------------------------------------------------
# Training (single target run)
# -----------------------------------------------------------------------------

def train_distill(
    config: RootConfig,
    args: argparse.Namespace,
    target, ## --> target
    tokenizer, 
    text_encoder, 
    unet, 
    noise_scheduler, 
    pipe,
    domain_map,
    amp_dtype: Optional[torch.dtype],
) -> None:
    # Setup logging metadata
    metadata = {"config": config.json()}
    model_metadata = {"prompts": target, "rank": str(config.network.rank), "alpha": str(config.network.alpha)}
    save_path = Path(config.save.path)

    weight_dtype = config_pkg.parse_precision(config.train.precision)
    save_weight_dtype = config_pkg.parse_precision(config.train.precision)

    # wandb
    if config.logging.use_wandb:
        project_name = f"{config.logging.project_name}"
        project_name += f"_last_token_conf{args.conf}"

        run_name = f"{config.logging.run_name}_depth{args.depth}_distill_noise_{args.noise_type}"

        wandb.init(
            project=project_name,
            config=metadata,
            name=run_name,
            settings=wandb.Settings(symlink=False),
        )

    # Build network
    network = build_network(unet, text_encoder, config, args, weight_dtype)
    network_teacher = build_network_teacher(unet, text_encoder, config, args, weight_dtype, arch_type="ffn")
    network_teacher.eval()
    network_teacher.set_inference_mode()
    network_teacher.requires_grad_(False)

    # Measurements & mappings
    lipschitz = compute_lipschitz_svals(unet)
    mapped_unet_modules = map_network_to_unet_modules(network, unet)


    noise_dict = load_or_make_noise(args.noise_type, mapped_unet_modules, save_path, rand_scale=args.rand)

    # AMP setup
    amp_enabled = amp_dtype is not None
    scaler = GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))


    # 2) Loader
    concepts_target = target
    in_dir = f"./dataset/train_pairs/conf{args.conf}"
    out_dir = f"./dataset/train_shards/conf{args.conf}"

    shard_paths = build_global_shuffled_shards(
        targets=concepts_target,
        in_dir=in_dir,
        out_dir=out_dir,
        domain_map=domain_map,
        num_shards=args.num_shards,
        n_batch=args.dataset_n_batch,
        total_iterations=args.iterations,
        seed=args.dataset_seed,
    )

    dataset = GlobalShuffledShardDataset(
        shard_paths=shard_paths,
        n_batch=args.dataset_n_batch,
        n_anc=args.dataset_n_anc,
        total_iterations=args.iterations,
        seed=args.dataset_seed,
        shuffle_within_shard=True,
        drop_last=True,
        anchors_path=os.path.join(out_dir, "anchors.pt"), 
    )

    data_loader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)



    # Optimizer / scheduler
    trainable = network.prepare_optimizer_params(config.train.text_encoder_lr, config.train.unet_lr, config.train.lr)
    network.requires_grad_(True).train()


    _opt_name, _opt_args, optimizer = get_optimizer(config, trainable)
    config.logging.interval = args.iterations
    config.train.iterations = args.iterations
    lr_scheduler = get_scheduler_fix(config, optimizer)
    criteria = torch.nn.MSELoss()


    # Progress bar & train (delegates to project trainer)
    pbar = tqdm(range(args.iterations))
    network = train_erase_one_stage(
        args,
        stage=0,
        pbar=pbar,
        config=config,
        device_cuda=DEVICE_CUDA,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        network=network,
        network_teacher=network_teacher,
        network_modules=map_network_to_unet_modules(network, unet),
        unet_modules=map_network_to_unet_modules(network, unet),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lipschitz=lipschitz,
        save_weight_dtype=save_weight_dtype,
        model_metadata={"prompts": "all", "rank": str(config.network.rank), "alpha": str(config.network.alpha)},
        data_loader=data_loader,
        save_path=save_path,
        amp_dtype=amp_dtype,
        amp_enabled=(amp_dtype is not None),
        scaler=scaler,
        noise_dict=noise_dict,
    )








    # Save weights
    print("Saving...")
    model_metadata={"prompts": "all", "rank": str(config.network.rank), "alpha": str(config.network.alpha)}
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(save_path / f"{config.save.name}_last.safetensors", dtype=save_weight_dtype, metadata=model_metadata)

    # Cleanup
    del unet, noise_scheduler, optimizer, network
    flush()
    if config.logging.use_wandb:
        wandb.finish()
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
    config.train.skip_learned = args.skip_learned

    # Experiment naming
    # Uses the *first* prompt for guide replacement; safe since we reassign in loop below too
    exp_name = (
        config.save.path.split("/")[-2]
        .replace("guide#", f"guide{args.guidance_scale}")
        .replace("pal#", f"pal{config.train.pal}")
        .replace("gate_rank#", f"gate_rank{config.network.init_size}")
    )

    exp_name += f"_depth{args.depth}_last_token_conf{args.conf}"
    exp_name += f"_rank{config.network.rank}"
    exp_name += f"_noise_{args.noise_type}"

    config.save.path = "/".join(config.save.path.split("/")[:-2] + [exp_name] + [config.save.path.split("/")[-1]])



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    


def main(args: argparse.Namespace) -> None:
    config = config_pkg.load_config_from_yaml(args.config_file)

    config.replace_word = config.replace_word.split(",")

    domains = ["celeb", "char", "artist"]
    domain_names = ["actor", "character", "style"]


    prompts = []
    domain_map = dict()
    for rep_word, d_name in zip(domains,domain_names):
        prompt_file = config.prompts_file.replace("hetero", rep_word)
        prompts_domain = prompt_pkg.load_prompts(prompt_file)
        prompts += prompts_domain
        for prmpt in prompts_domain:
            domain_map[prmpt] = d_name 

    
    update_config_from_args(config, args)

    base_path = config.save.path

    # AMP policy
    amp_dtype = select_amp_dtype(getattr(args, "amp_dtype", "auto"))

    # Components
    weight_dtype = config_pkg.parse_precision(config.train.precision)
    tokenizer, text_encoder, unet, noise_scheduler, pipe = load_sd_components(config)
    enable_eval_fwd_only(unet, text_encoder, weight_dtype)
    pipe.safety_checker = None
    set_matmul_precision()

    # Logging & save path per-target
    os.makedirs(config.save.path, exist_ok=True)

    print(f"Num target={len(prompts)} | pal={config.train.pal}")

    seed_everything(config.train.train_seed)

    train_distill(config, args, prompts, tokenizer, text_encoder, unet, noise_scheduler, pipe, domain_map, amp_dtype=amp_dtype)


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
    parser.add_argument("--model_path", required=True, nargs="*", help="EAS model to use.")

    parser.add_argument("--use_bias", type=str2bool)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--dataset_n_batch", type=int, default=64)
    parser.add_argument("--dataset_n_anc", type=int, default=4)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=-1)
    parser.add_argument("--iters_per_target", type=int, default=-1)

    parser.add_argument("--model_domains", required=True, nargs="*", help="EAS model to use.")

    parser.add_argument("--noise_type", type=str, default="row_wise")


    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="auto",  # auto|bf16|fp16|off
        help="Mixed precision dtype: auto(bf16 if supported, else fp16), bf16, fp16, off",
    )

    args = parser.parse_args()
    main(args)
