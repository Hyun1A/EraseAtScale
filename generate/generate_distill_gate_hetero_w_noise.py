import argparse
import gc
from pathlib import Path
import pandas as pd
import torch
from typing import Literal
import sys
import random, os
import numpy as np
from safetensors.torch import load_file

sys.path[0] = "/".join(sys.path[0].split('/')[:-1])

from src.configs.generation_config import load_config_from_yaml, GenerationConfig
from src.configs.config import parse_precision
from src.engine import train_util
from src.models import model_util
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

from src.models.merge_eas import *
device = torch.device('cuda:0')
torch.cuda.set_device(device)

UNET_NAME = "unet"
TEXT_ENCODER_NAME = "text_encoder"



def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def flush():
    torch.cuda.empty_cache()
    gc.collect()



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





def infer_with_eas(
        args,
        model_paths: list[str],
        config: GenerationConfig,
        base_model: str = "CompVis/stable-diffusion-v1-4",
        v2: bool = False,
        precision: str = "fp32",
    ):

    weight_dtype = parse_precision(precision)
        
    # load the pretrained SD
    tokenizer, text_encoder, unet, pipe = model_util.load_checkpoint_model(
        base_model,
        v2=v2,
        weight_dtype=weight_dtype
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    special_token_ids = set(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values()))

    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    eas_modules, metadatas = zip(*[
        load_state_dict(model_path, weight_dtype) for model_path in model_paths
    ])
    
    # check if EASs are compatible
    assert all([metadata["rank"] == metadatas[0]["rank"] for metadata in metadatas])

    # get the erased concept
    erased_prompts = [md["prompts"].split(",") for md in metadatas]
    erased_prompts_count = [len(ep) for ep in erased_prompts]
    print(f"Erased prompts: {erased_prompts}")


    if args.arch_type == "gate":
        arch = EASLayer_Gate
    elif args.arch_type == "mlp":
        arch = EASLayer_MLP
    elif args.arch_type == "mlp_swiglu":
        arch = EASLayer_MLP_SwiGLU
    elif args.arch_type == "linear":
        arch = EASLayer_Linear
    elif args.arch_type == "ffn":
        arch = EASLayer_FFN
    elif args.arch_type == "moe_dense":
        arch = EASLayer_MoE_Dense


    module_kwargs = _collect_moe_kwargs_from_args(args) if args.arch_type.lower() == "moe" else {}



    network = EASNetwork(
        unet,
        text_encoder,
        rank=int(float(metadatas[0]["rank"])),
        multiplier=1.0,
        alpha=float(metadatas[0]["alpha"]),
        module=arch,
        module_kwargs=module_kwargs,
        continual=True,
        task_id=10,
        continual_rank=config.gate_rank, 
        hidden_size=config.gate_rank,  
        init_size=config.gate_rank,  
        n_concepts=len(model_paths),
        args=args,
    ).to(device, dtype=weight_dtype)  


    if args.arch_type == "gate":
        for k,v in network.named_parameters(): 
            print(f"{k:100}", v.shape, eas_modules[0][k].shape)
            for idx in range(len(eas_modules)):
                # try:
                if len(v.shape) > 1:
                    v.data[idx,:] = eas_modules[idx][k]
                elif args.arch_type=="gate":
                    v.data[idx] = eas_modules[idx][k]
                else:
                    v.data = eas_modules[idx][k]
    else:
        network.load_state_dict(eas_modules[0])
    
    network.to(device, dtype=weight_dtype)  


    network_modules = dict()
    for name, module in network.named_modules():
        if "EASLayer" in module.__class__.__name__:
            network_modules[name] = module

    unet_modules = dict()
    for name, module in unet.named_modules():
        name = "_".join(name.split("."))
        name = "lora_unet_" + name

        for network_name in network_modules.keys():
            if name == network_name:
                unet_modules[name] = module    




    ####################################
    ############ add noise #############
    path_comps = model_paths[0]._str.split("/")[:-2]+["noise_low_rank_1.safetensors"]
    noise_dict = load_file("/".join(path_comps))
    for key, val in noise_dict.items():
        noise_dict[key] = val.to(device, dtype=weight_dtype)

    for key, val in unet_modules.items():
        weight = val.weight.data
        weight_noise = noise_dict[key]
        unet_modules[key].weight.data = (weight + weight_noise).clone()
    ############ add noise #############
    ####################################
    
    network.eval()
    network.set_inference_mode()
    
    print(config.save_path)
    
    promptDf = pd.read_csv(config.prompt_path)
    with torch.no_grad():
        for p_idx, (k, row) in enumerate(promptDf.iterrows()):
            
            if (p_idx < config.st_prompt_idx) or (p_idx > config.end_prompt_idx):
                continue
            
            prompt = row['prompt']
            prompt += config.unconditional_prompt
            print(f"{p_idx}, Generating for prompt: {prompt}")
            prompt_embeds, prompt_tokens = train_util.encode_prompts(
                tokenizer, text_encoder, [prompt], return_tokens=True
                )
            
            print(os.path.isfile(config.save_path.format(prompt.replace(" ", "_"), row['evaluation_seed'], "0")))
            if os.path.isfile(config.save_path.format(prompt.replace(" ", "_"), row['evaluation_seed'], "0")):
                print(row)
                print('cont')  
                continue
                
            network.reset_cache_attention_gate()

            seed_everything(row['evaluation_seed'])
            
            with network:
                images = pipe(
                    negative_prompt=config.negative_prompt,
                    width=config.width,
                    height=config.height,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=torch.cuda.manual_seed(row['evaluation_seed']),
                    num_images_per_prompt=config.generate_num,
                    prompt_embeds=prompt_embeds,
                ).images


            folder = Path(config.save_path.format(prompt.replace(" ", "_"), "0", "0")).parent
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)
            for i, image in enumerate(images):
                image.save(
                    config.save_path.format(
                        prompt.replace(" ", "_"), row['evaluation_seed'], i
                    )
                )
            
            network.reset_cache_attention_gate()




def main(args):

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

    model_path = [Path(lp) for lp in concepts_ckpt]
    
    generation_config = load_config_from_yaml(args.config)

    if args.st_prompt_idx != -1:
        generation_config.st_prompt_idx = args.st_prompt_idx
    if args.end_prompt_idx != -1:
        generation_config.end_prompt_idx = args.end_prompt_idx
    if args.gate_rank != -1:
        generation_config.gate_rank = args.gate_rank
    
    
    generation_config.save_path = os.path.join("/".join(generation_config.save_path.split("/")[:-3]), args.save_env, "/".join(generation_config.save_path.split("/")[-2:]))

    infer_with_eas(
        args,
        model_path,
        generation_config,
        base_model=args.base_model,
        v2=args.v2,
        precision=args.precision,
    )




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/generation.yaml",
        help="Base configs for image generation.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        nargs="*",
        help="EAS model to use.",
    )

    parser.add_argument(
        "--model_domains",
        required=True,
        nargs="*",
        help="EAS model to use.",
    )

    # model configs
    parser.add_argument(
        "--base_model",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Base model for generation.",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use the 2.x version of the SD.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Precision for the base model.",
    )
    parser.add_argument(
        "--save_env",
        type=str,
        default="",
        help="Precision for the base model.",
    )    
    
    parser.add_argument(
        "--st_prompt_idx",
        type=int,
        default=-1,
    )
    
    parser.add_argument(
        "--end_prompt_idx",
        type=int,
        default=-1,
    )
    
    parser.add_argument(
        "--gate_rank",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--rand",
        type=float,
        default=0.01,
    )


    parser.add_argument(
        "--arch_type",
        type=str,
        default="gate",
    )


    parser.add_argument(
        "--use_bias",
        type=str2bool,
    )

    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
    )

    
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--concept",
        type=str,
        default="none",
    )


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


    args = parser.parse_args()

    main(args)
