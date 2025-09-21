# ref: 
# - https://github.com/p1atdev/LECO/blob/main/train_util.py

from typing import Optional, Union
import os, sys
import ast
import importlib
import math
import pandas as pd
import random
import numpy as np
import gc
from tqdm import tqdm
import torch
from diffusers import LMSDiscreteScheduler




def flush():
    torch.cuda.empty_cache()
    gc.collect()


def get_module_name_type(find_module_name):
    if find_module_name == "unet_ca":
        module_type = "Linear"
        module_name = "attn2"

    elif find_module_name == "unet_ca_kv":
        module_type = "Linear"
        module_name = "attn2"

    elif find_module_name == "unet_ca_v":
        module_type = "Linear"
        module_name = "attn2"
        

    elif find_module_name == "unet_ca_out":
        module_type = "Linear"
        module_name = "attn2"
        
    elif find_module_name == "unet_sa_out":
        module_type = "Linear"
        module_name = "attn1"

    elif find_module_name == "unet_sa":
        module_type = "Linear"
        module_name = "attn1"

    elif find_module_name == "unet_conv2d":
        module_type = "Conv2d"
        module_name = "conv2d"           

    elif find_module_name == "unet_misc":
        module_type = "Linear"
        module_name = "misc"

    elif find_module_name == "te_attn":        
        module_type = "Linear"
        module_name = "self_attn"

    else:
        module_type = "Linear"
        module_name = "mlp.fc"

    return module_name, module_type

def get_modules_list(unet, text_encoder, find_module_name, module_name, module_type):
    org_modules = dict()
    module_name_list = []

    if find_module_name == "unet_ca_out":
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if (module_name+".to_out" in n):
                    module_name_list.append(n)
                    org_modules[n] = m

    elif find_module_name == "unet_ca_kv":
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if (module_name+".to_k" in n) or (module_name+".to_v" in n):
                    module_name_list.append(n)
                    org_modules[n] = m


    elif find_module_name == "unet_ca_v":
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if (module_name+".to_v" in n):
                    module_name_list.append(n)
                    org_modules[n] = m
                    

    elif find_module_name == "unet_sa_out":
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if (module_name+".to_out" in n):
                    module_name_list.append(n)
                    org_modules[n] = m

    elif "unet" in find_module_name:
        for n,m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if module_name == "misc":
                    if ("attn1" not in n) and ("attn2" not in n):
                        module_name_list.append(n)
                        org_modules[n] = m

                elif (module_name == "attn1") or (module_name == "attn2"): 
                    if module_name in n:
                        module_name_list.append(n)
                        org_modules[n] = m

                else:
                    module_name_list.append(n)
                    org_modules[n] = m

    else:
        for n,m in text_encoder.named_modules():
            if m.__class__.__name__ == module_type:       
                if module_name in n:
                    module_name_list.append(n)
                    org_modules[n] = m

    return org_modules, module_name_list

def load_model_sv_cache(find_module_name, param_cache_path, device, org_modules):
    
    if os.path.isfile(f"{param_cache_path}/vh_cache_dict_{find_module_name}.pt"):
        print("load precomputed svd for original models ....")

        param_vh_cache_dict = torch.load(f"{param_cache_path}/vh_cache_dict_{find_module_name}.pt", map_location=torch.device(device)) 
        param_s_cache_dict = torch.load(f"{param_cache_path}/s_cache_dict_{find_module_name}.pt", map_location=torch.device(device))

    else:
        print("compute svd for original models ....")

        param_vh_cache_dict = dict()
        param_s_cache_dict = dict()

        for idx_mod, (k,m) in enumerate(org_modules.items()):
            print(idx_mod, k)
            if m.__class__.__name__ == "Linear":
                U,S,Vh = torch.linalg.svd(m.weight, full_matrices=False) 
                param_vh_cache_dict[k] = Vh.detach().cpu()
                param_s_cache_dict[k] = S.detach().cpu()

            elif m.__class__.__name__ == "Conv2d":
                module_weight_flatten = m.weight.view(m.weight.size(0), -1)

                U,S,Vh = torch.linalg.svd(module_weight_flatten, full_matrices=False)
                param_vh_cache_dict[k] = Vh.detach().cpu()
                param_s_cache_dict[k] = S.detach().cpu()                

        os.makedirs(param_cache_path, exist_ok=True)
        torch.save(param_vh_cache_dict, f"{param_cache_path}/vh_cache_dict_{find_module_name}.pt")
        torch.save(param_s_cache_dict, f"{param_cache_path}/s_cache_dict_{find_module_name}.pt")

    return param_vh_cache_dict, param_s_cache_dict




def get_registered_buffer_importance(args, module_name_list_all, org_modules_all, st_timestep, \
                        end_timestep, n_avail_tokens, prompts, embeddings, embeddings_surr, embedding_uncond, \
                        pipe, device, register_buffer_path, register_buffer_fn, register_func, **kwargs):



    embs_batchsize = 100
    embs_batch = []
    embs_surr_batch = []
    prompts_batch = []
    len_embs_batch = embeddings.size(0)


    os.makedirs(register_buffer_path, exist_ok=True)


    if os.path.isfile(f"{register_buffer_path}/{register_buffer_fn}"):
        print(f"load precomputed registered_buffer for original models ... {register_buffer_path}/{register_buffer_fn}")
        importance_diag = torch.load(f"{register_buffer_path}/{register_buffer_fn[:-3]+'_importance.pt'}", map_location=torch.device(device))

    else:
        print(f"compute registered_buffer for original models ... {register_buffer_path}/{register_buffer_fn}")
        for batch_idx in range(int(math.ceil(float(len_embs_batch)/embs_batchsize))):
            if embs_batchsize*(batch_idx+1) <= len_embs_batch:
                embs_batch.append(embeddings[embs_batchsize*batch_idx:embs_batchsize*(batch_idx+1)])
                embs_surr_batch.append(embeddings_surr[embs_batchsize*batch_idx:embs_batchsize*(batch_idx+1)])
                prompts_batch.append(prompts[embs_batchsize*batch_idx:embs_batchsize*(batch_idx+1)])
            
            else:
                embs_batch.append(embeddings[embs_batchsize*batch_idx:])
                embs_surr_batch.append(embeddings_surr[embs_batchsize*batch_idx:])
                prompts_batch.append(prompts[embs_batchsize*batch_idx:])

        for step, (embs, embs_surr, prompts) in enumerate(zip(embs_batch, embs_surr_batch, prompts_batch)):

            if step % 10 == 0:
                print(f"{step}/{len(embs_batch)}")

            if not ( step >= 0 and step < 125 ):
                continue


            for seed in range(3):

                if len(embs.size())==4:
                    B,C,T,D = embs.size()
                    embs = embs.reshape(B*C,T,D)

                if "save_path" in kwargs.keys():
                    save_path = f"{kwargs['save_path']}/seed_{seed}"
                    os.makedirs(f"{save_path}", exist_ok=True)
                    save_path = f"{save_path}/image.png"

                else:
                    save_path = "./test2.png"

                importance_diag = get_importance(prompts, module_name_list_all, embs, embs_surr, "", pipe, seed=seed, \
                                    uncond_embeddings=embedding_uncond, end_timestep=end_timestep, save_path=save_path)
                

        torch.save({prompts[0][0]:importance_diag}, f"{register_buffer_path}/{register_buffer_fn[:-3]+'_importance.pt'}")

    return importance_diag




# -----------------------------------------------------------------------------
# for computing importance
# -----------------------------------------------------------------------------


def get_importance(prompts, module_name_list_all, embeddings, embeddings_surr, \
                df, ldm_stable, save_path="./", save_name=None, device='cuda:0', guidance_scale = 7.5, image_size=512, \
                ddim_steps=50, num_samples=1, from_case=0, to_case=None,seed=0, target="", uncond_embeddings=None, end_timestep=-1):
        

    vae = ldm_stable.vae
    tokenizer = ldm_stable.tokenizer
    text_encoder = ldm_stable.text_encoder
    unet = ldm_stable.unet

    importance_diag = dict()
    unet_parameters_org = dict()
    for key, mod in ldm_stable.unet.named_modules():
        if key in module_name_list_all[0]:
            unet_parameters_org[key] = mod.weight.clone()
            importance_diag[key] = torch.zeros_like(mod.weight).to(mod.weight.device)

    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    folder_path = './generated_images'

    
    height,width = image_size, image_size
    num_inference_steps = ddim_steps          

    generator = torch.cuda.manual_seed(seed)        # Seed generator to create the inital latent noise

    batch_size = len(embeddings)

    if uncond_embeddings is None:
        max_length = embeddings.shape[1]
        
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    else:
        uncond_embeddings = uncond_embeddings.repeat(batch_size,1,1)
    
    text_embeddings_cat = torch.cat([uncond_embeddings.repeat(1,1,1), \
                                    embeddings], dim=0)


    latents = torch.randn(
        (1, unet.config.in_channels, height // 8, width // 8),
        generator=generator,device=torch_device
    ).repeat(batch_size,1,1,1)
    latents = latents.to(device=torch_device, dtype=torch.bfloat16)
    
    loss_org, noise_pred_map_org_list = compute_loss(unet, latents.clone(), num_inference_steps, guidance_scale, \
                    text_embeddings_cat, importance_diag, scheduler, batch_size, end_timestep)


    scale = 0.001
    for idx in range(5):
        #   1. precompute perturbations, setting noise scale
        noise_dict = dict()
        generator=torch.Generator(device=latents.device).manual_seed(idx)
        for key, val in unet_parameters_org.items():
            if len(val.size()) == 1:
                noise_dict[key] = scale * torch.randn(val.size(),device=torch_device, generator=generator).to(torch.bfloat16)
            elif len(val.size()) == 2:
                noise_dict[key] = scale * torch.randn(val.size(),device=torch_device, generator=generator).to(torch.bfloat16)


        # for key, val in unet_parameters_org.items():
        #     if len(val.size()) == 1:
        #         noise_dict[key] = scale * val.norm() * torch.randn_like(val) / val.size(0)
        #     elif len(val.size()) == 2:
        #         noise_dict[key] = scale * val.norm() * torch.randn_like(val) / (val.size(0)*val.size(1))**0.5


        #   2. inject noise to targeted parameters
        for key, mod in unet.named_modules():
            if key in module_name_list_all[0]:
                mod.weight.data = (unet_parameters_org[key] + noise_dict[key]).clone()

        #   3. stack loss for timesteps
        loss, _ = compute_loss(unet, latents.clone(), num_inference_steps, guidance_scale, \
            text_embeddings_cat, importance_diag, scheduler, batch_size, end_timestep, \
            noise_pred_map_org_list=noise_pred_map_org_list)

        for key, mod in unet.named_modules():
            if key in module_name_list_all[0]:
                mod.weight.data = unet_parameters_org[key].clone()
                
        #   4. compute difference
        loss_diff = loss - loss_org
        scale_diff = loss_diff / scale
        
        #   5. stack conceptwise importance value
        for key, val in importance_diag.items():
            importance_diag[key] += scale_diff * noise_dict[key]

    for key, val in importance_diag.items():
        importance_diag[key] = importance_diag[key]**2

    return importance_diag



def compute_loss(unet, latents, num_inference_steps, guidance_scale, text_embeddings_cat, \
                importance_diag, scheduler, batch_size, end_timestep, noise_pred_map_org_list=None):
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps)

    loss = None

    if noise_pred_map_org_list is None:
        noise_pred_map_org_comp = []

    for idx, t in enumerate(tqdm(scheduler.timesteps)):

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        if importance_diag is not None:
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_cat).sample
        else:
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_cat).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        # breakpoint()
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        noise_pred_target, noise_pred_map = noise_pred, noise_pred

        step_index = (scheduler.timesteps == t).nonzero().item()
        sigma = scheduler.sigmas[step_index]

        if noise_pred_map_org_list is not None:
            noise_pred_map = noise_pred_map_org_list[idx]
        loss_step = sigma*(( noise_pred_target - noise_pred_map )**2).mean(dim=1).mean(dim=1).mean(dim=1)

        loss = loss_step if loss is None else loss+loss_step

        if noise_pred_map_org_list is None:
            noise_pred_map_org_comp.append(noise_pred_map)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample.detach()

        if idx==end_timestep:
            break

    if noise_pred_map_org_list is None:
        return loss, noise_pred_map_org_comp
    else: 
        return loss, noise_pred_map_org_list
