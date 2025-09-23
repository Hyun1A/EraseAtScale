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

torch.set_float32_matmul_precision("high")

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
    elif name == "mlp":
        return EASLayer_MLP
    elif name == "mlp_swiglu":
        return EASLayer_MLP_SwiGLU
    elif name == "linear":
        return EASLayer_Linear
    elif name == "ffn":
        return EASLayer_FFN
    elif name == "ffn_glue":
        return EASLayer_FFN_GLU
    elif name == "moe_dense":
        return EASLayer_MoE_Dense
    raise ValueError(f"Unsupported arch_type: {name}")



def _collect_moe_kwargs_from_args(args: argparse.Namespace) -> dict:
    return dict(
        num_experts      = int(getattr(args, "n_experts", 4)),
        top_k            = int(getattr(args, "top_k", 2)),
        router_noise_std = float(getattr(args, "router_noise_std", 0.0)),
        keeptok          = float(getattr(args, "keeptok", 0.0)),
        lb_coef          = float(getattr(args, "lb_coef", 1e-2)),
        cov_coef         = float(getattr(args, "cov_coef", 1e-3)),
        ffn_norm         = str(getattr(args, "ffn_norm", "rmsnorm")),
        dropout_p        = float(getattr(args, "moe_dropout", 0.0)),
        resid_dropout_p  = float(getattr(args, "moe_resid_dropout", 0.0)),
        prenorm_router   = bool(getattr(args, "prenorm_router", True)),
        use_bias         = getattr(args, "use_bias", None),
        depth            = getattr(args, "depth", None),
        sparse_compute   =True,
        glu_type = str(getattr(args, "glu_type", "swi")),
    )


def build_network(
    unet,
    text_encoder,
    config: RootConfig,
    args: argparse.Namespace,
    weight_dtype: torch.dtype,
):
    task_id = len(config.pretrained_model.safetensor)
    arch = pick_arch(args.arch_type)

    module_kwargs = _collect_moe_kwargs_from_args(args) if "moe" in args.arch_type.lower() else {}
    
    net = EASNetwork(
        unet,
        text_encoder,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        module=arch,
        module_kwargs=module_kwargs,
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

    if arch_type == "gate":
        net_type_org = args.net_type
        args.net_type = "ca_kv"

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

    if arch_type == "gate":
        args.net_type = net_type_org

        for k,v in net.named_parameters(): 
            print(f"{k:100}", v.shape, eas_modules[0][k].shape)
            for idx in range(len(eas_modules)):
                if len(v.shape) > 1:
                    v.data[idx,:] = eas_modules[idx][k]
                else:
                    v.data[idx] = eas_modules[idx][k]
    else:
        net.load_state_dict(eas_modules[0])

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
    os.makedirs("./output/noise", exist_ok=True)

    if  noise_type == "row_wise":
        noise_path = Path(f"./output/noise/noise_{noise_type}.safetensors")
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
        noise_path = Path(f"./output/noise/noise_{noise_type}.safetensors")
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
        project_name += f"_conf{args.conf}"

        run_name = f"{config.logging.run_name}_{args.net_type}_{args.arch_type}_{args.glu_type}"
        run_name += f"_d{args.depth}_E{args.n_experts}_k{args.top_k}_keep{args.keeptok}"
        run_name += f"_b{args.dataset_n_batch}_lr{config.train.lr}_it{args.iterations}"
        run_name += f"_map_{args.mapping_type}_{args.n_top}"

        if args.noise_type != "none":
            run_name += f"_noise_{args.rand}_{args.noise_type}"

        wandb.init(
            project=project_name,
            config=metadata,
            name=run_name,
            settings=wandb.Settings(symlink=False),
        )

    # Build network
    network = build_network(unet, text_encoder, config, args, weight_dtype)
    # network = torch.compile(network, mode='reduce-overhead')

    # Measurements & mappings
    lipschitz = compute_lipschitz_svals(unet)
    mapped_unet_modules = map_network_to_unet_modules(network, unet)
    for k,v in lipschitz.items():
        lipschitz[k] = [dat.to(torch.bfloat16) for dat in v]

    noise_dict = load_or_make_noise(args.noise_type, mapped_unet_modules, save_path, rand_scale=args.rand) \
                if args.noise_type != "none" else None

    if noise_dict is None:
        network_teacher = build_network_teacher(unet, text_encoder, config, args, weight_dtype, arch_type="gate")
    else:
        network_teacher = build_network_teacher(unet, text_encoder, config, args, weight_dtype, arch_type=args.arch_type)
    
    unet.to(torch.bfloat16)
    text_encoder.to(torch.bfloat16)
    network.to(torch.bfloat16)

    network_teacher.eval()
    network_teacher.set_inference_mode()
    network_teacher.requires_grad_(False)
    network_teacher.to(torch.bfloat16)

    # AMP setup
    amp_enabled = amp_dtype is not None
    scaler = GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

    # breakpoint()

    # 2) Loader
    concepts_target = target
    in_dir = f"./dataset/train_pairs/{args.mapping_type}_{args.n_top}/conf{args.conf}"
    out_dir = f"./dataset/train_shards/{args.mapping_type}_{args.n_top}/conf{args.conf}"

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
    exp_name += f"_map_{args.mapping_type}_{args.n_top}"

    if "moe" in args.arch_type.lower():
        exp_name += f"_{args.net_type.lower()}"
        exp_name += f"_{args.arch_type.lower()}_{args.glu_type}_E{args.n_experts}_k{args.top_k}_keep{args.keeptok}"
        exp_name += f"_b{args.dataset_n_batch}_lr{config.train.lr}_it{args.iterations}"

    if args.noise_type != "none":
        exp_name += f"_noise_{args.rand}_{args.noise_type}"

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

    prompts_available = []
    for target in prompts:    
        domain = domain_map[target]
        if os.path.isfile(f"./dataset/train_pairs/{args.mapping_type}_{args.n_top}/conf{args.conf}/{domain}/{target}.pt"):
            prompts_available.append(target)

    print(f"Num target available={len(prompts_available)} | pal={config.train.pal}")

    seed_everything(config.train.train_seed)

    train_distill(config, args, prompts_available, tokenizer, text_encoder, unet, noise_scheduler, pipe, domain_map, amp_dtype=amp_dtype)


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
    parser.add_argument("--arch_type", type=str, default="gate")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.9)
    parser.add_argument("--model_path", required=True, nargs="*", help="EAS model to use.")

    parser.add_argument("--use_bias", type=str2bool)
    parser.add_argument("--activation", type=str, default="none")
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--dataset_n_batch", type=int, default=64)
    parser.add_argument("--dataset_n_anc", type=int, default=4)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=-1)
    parser.add_argument("--iters_per_target", type=int, default=-1)
    parser.add_argument("--model_domains", required=True, nargs="*", help="EAS model to use.")

    parser.add_argument("--n_experts", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--router_noise_std", type=float, default=0.0)
    parser.add_argument("--keeptok", type=float, default=0.0)          # 0~1
    parser.add_argument("--lb_coef", type=float, default=1e-2)
    parser.add_argument("--cov_coef", type=float, default=1e-3)
    parser.add_argument("--ffn_norm", type=str, default="rmsnorm")      # layernorm|rmsnorm|scalenorm
    parser.add_argument("--moe_dropout", type=float, default=0.0)
    parser.add_argument("--moe_resid_dropout", type=float, default=0.0)
    parser.add_argument("--prenorm_router", type=str2bool, default=True)
    parser.add_argument("--moe_aux_coef", type=float, default=1.0)  
    parser.add_argument("--sparse_compute", type=str2bool, default=True)
    parser.add_argument("--capacity_factor", type=float, default=1.25)
    parser.add_argument("--token_drop_policy", type=str, default="bypass")  # bypass|zero
    parser.add_argument("--glu_type", type=str, default="swi")  # "swi" or "glu"
    parser.add_argument("--net_type", type=str, default="ca_kv")  
    parser.add_argument("--lr", type=float, default=-1) 
    parser.add_argument("--mapping_type", type=str, default="base") 
    parser.add_argument("--n_top", type=int, default=3)
    parser.add_argument("--rand", type=float, default=0.5)
    parser.add_argument("--noise_type", type=str, default="none")

    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="auto",  # auto|bf16|fp16|off
        help="Mixed precision dtype: auto(bf16 if supported, else fp16), bf16, fp16, off",
    )

    args = parser.parse_args()
    main(args)
