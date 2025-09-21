from pathlib import Path
import gc
from copy import deepcopy
import pandas as pd
import random

import torch
from tqdm import tqdm
import os, sys
import numpy as np
from torch.cuda.amp import autocast

sys.path[0] = "/".join(sys.path[0].split('/')[:-1])
import src.engine.train_util as train_util

from contextlib import contextmanager
from torch.cuda.amp import autocast

import wandb

def train_erase_one_stage(
        args,
        stage,
        pbar,
        config,
        device_cuda,
        tokenizer,
        text_encoder,
        network,
        network_teacher,
        network_modules,
        unet_modules,
        optimizer,
        lr_scheduler,
        lipschitz,
        save_weight_dtype,
        model_metadata,
        data_loader,
        save_path: Path | None = None,
        amp_dtype=None,
        amp_enabled=False,
        scaler=None,
        noise_dict=None,
    ):

    embedding_unconditional = train_util.encode_prompts(tokenizer, text_encoder, [""])
    data_iter = iter(data_loader)
    g_scale = args.guidance_scale
    pal = torch.tensor([config.train.pal]).float().to(device=device_cuda)     

    for i in pbar:
        loss=dict()
        optimizer.zero_grad(set_to_none=True)

        
        #####################################################################
        ################### Prepare training embeddings #####################

        try:
            targets, mapping, anchors = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            targets, mapping, anchors = next(data_iter)

        targets = targets.to(device_cuda, non_blocking=True).float()
        mapping = mapping.to(device_cuda, non_blocking=True).float()
        anchors = anchors.to(device_cuda, non_blocking=True).float()
        
        if len(anchors.size()) == 4:
            a,b,c,d = anchors.size()
            anchors = anchors.reshape(a*b,c,d)


        if len(targets.size()) == 2:
            targets = targets[None,:,:]
            mapping = mapping[None,:,:]
        
        b,t,d = targets.size()
        if i == 0:
            uncond = embedding_unconditional[:, :t].repeat(b,1,1)
        # targets = targets.reshape(-1,d)[:,None,:]
        # mapping = mapping.reshape(-1,d)[:,None,:]
        # anchors = anchors.reshape(-1,d)[:,None,:]
        n_pairs = targets.size(0)


        ################### Prepare training embeddings #####################
        #####################################################################        
        
        #########################################################
        ############### loss_prompt_erase/anchor ################   

        with autocast(dtype=amp_dtype, enabled=amp_enabled):
            pal_k_coef_log_dict_erase = dict()
            pal_v_coef_log_dict_erase = dict()        
            loss_prompt_erase_to_k = 0
            loss_prompt_erase_to_v = 0

            pal_k_coef_log_dict_anchor = dict()
            pal_v_coef_log_dict_anchor = dict()
            loss_prompt_anchor_to_k = 0
            loss_prompt_anchor_to_v = 0    
            
            idx = 0

            cat_inputs = torch.cat([uncond, targets, mapping, anchors], dim=0).float()
        
            for name in network_modules.keys():
                if "lora_unet" not in name or "lora_adaptor" in name:
                    continue


                with torch.no_grad():
                    with network_teacher.editing():
                        crsattn_org = unet_modules[name](cat_inputs)
                        crsattn_target_org = crsattn_org[n_pairs:][:n_pairs]
                        crsattn_neutral_org = crsattn_org[n_pairs:][n_pairs:2*n_pairs]
                        crsattn_comp_org = torch.cat([crsattn_org[:n_pairs], crsattn_org[n_pairs:][2*n_pairs:]], dim=0)

                if noise_dict is not None:
                    weight = unet_modules[name].weight.data
                    weight_noise = noise_dict[name]
                    unet_modules[name].weight.data = (weight + weight_noise).clone()


                with network.editing():
                    crsattn = unet_modules[name](cat_inputs)
                    crsattn_target = crsattn[n_pairs:][:n_pairs]
                    crsattn_comp = torch.cat([crsattn[:n_pairs], crsattn[n_pairs:][2*n_pairs:]], dim=0)

                    
                if "to_k" in name:
                    lipschitz_for_key_target = ( lipschitz['lipschitz_ov'][idx]*lipschitz['lipschitz_q'][idx] ).unsqueeze(0).unsqueeze(1).unsqueeze(2) / 77
                    loss_prompt_erase_to_k += (lipschitz_for_key_target * (crsattn_neutral_org - crsattn_target)**2).mean()
                    pal_k_coef_log_dict_erase[f"pal_k_coef_log_dict_erase/{idx}th-layer"] = lipschitz_for_key_target.mean()

                    lipschitz_for_key_comp = ( lipschitz['lipschitz_ov'][idx]*lipschitz['lipschitz_q'][idx] ).unsqueeze(0).unsqueeze(1).unsqueeze(2) / 77
                    loss_prompt_anchor_to_k += (lipschitz_for_key_comp * (crsattn_comp_org-crsattn_comp)**2).mean()
                    pal_k_coef_log_dict_anchor[f"pal_k_coef_log_dict_anchor/{idx}th-layer"] = lipschitz_for_key_comp.mean()

                else:
                    lipschitz_for_val_target = lipschitz['lipschitz_o'][idx].unsqueeze(0).unsqueeze(1)
                    loss_prompt_erase_to_v += (lipschitz_for_val_target * (crsattn_neutral_org - crsattn_target)**2).mean()
                    pal_v_coef_log_dict_erase[f"pal_v_coef_log_dict_erase/{idx}th-layer"] = lipschitz_for_val_target.mean()

                    lipschitz_for_val_comp = lipschitz['lipschitz_o'][idx].unsqueeze(0).repeat(crsattn_comp.shape[0],1).unsqueeze(2)
                    loss_prompt_anchor_to_v += (lipschitz_for_val_comp * (crsattn_comp_org-crsattn_comp)**2).mean() #/ crsattn_comp_org.shape[0]
                    pal_v_coef_log_dict_anchor[f"pal_v_coef_log_dict_anchor/{idx}th-layer"] = lipschitz_for_val_comp.mean()                

                    idx+=1
            
            
            nmods = max(1, len(network_modules))

            loss_prompt_erase_to_k = loss_prompt_erase_to_k / nmods
            loss_prompt_erase_to_v = loss_prompt_erase_to_v / nmods
            loss_prompt_erase = loss_prompt_erase_to_v + loss_prompt_erase_to_k       

            loss_prompt_anchor_to_k = loss_prompt_anchor_to_k / nmods
            loss_prompt_anchor_to_v = loss_prompt_anchor_to_v / nmods
            loss_prompt_anchor = loss_prompt_anchor_to_v + loss_prompt_anchor_to_k        


            
            loss[f"loss_erasing_stage{stage}/loss_prompt_erase"] = loss_prompt_erase
            loss[f"loss_erasing_stage{stage}/loss_prompt_erase_to_k"] = loss_prompt_erase_to_k
            loss[f"loss_erasing_stage{stage}/loss_prompt_erase_to_v"] = loss_prompt_erase_to_v

            loss[f"loss_erasing_stage{stage}/loss_prompt_anchor"] = loss_prompt_anchor
            loss[f"loss_erasing_stage{stage}/loss_prompt_anchor_to_k"] = loss_prompt_anchor_to_k
            loss[f"loss_erasing_stage{stage}/loss_prompt_anchor_to_v"] = loss_prompt_anchor_to_v
            
            loss[f"loss_erasing"] = loss[f"loss_erasing_stage{stage}/loss_prompt_erase"] \
                                + pal * loss[f"loss_erasing_stage{stage}/loss_prompt_anchor"]

            loss_total = loss[f"loss_erasing"]

            ############### loss_prompt_erase/anchor ################        
            #########################################################                


            #########################################################        
            ######################### misc ##########################    
            loss["pal"] = pal
            loss["guidance"] = torch.tensor([g_scale]).cuda()
            ######################### misc ##########################        
            #########################################################        



        #########################################################        
        ######################## optim ##########################        
        
        # ----- AMP Optimization (Scaler) -----
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

        else:
            loss_total.backward()
            optimizer.step()
            lr_scheduler.step()

        if noise_dict is not None:
            for key, val in unet_modules.items():
                weight = val.weight.data
                weight_noise = noise_dict[key]
                unet_modules[key].weight.data = (weight - weight_noise).clone()


        ######################## optim ##########################        
        #########################################################

        #########################################################
        ####################### logging #########################     
        pbar.set_description(f"Loss: {loss[f'loss_erasing'].item():.4f}")
        
        if i%25 == 0 and config.logging.use_wandb:
            log_dict = {"iteration": i}
            loss = {k: v.detach().cpu().item() for k, v in loss.items()}
            log_dict.update(loss)
            lrs = lr_scheduler.get_last_lr()
            if len(lrs) == 1:
                log_dict["lr"] = float(lrs[0])
            else:
                log_dict["lr/textencoder"] = float(lrs[0])
                log_dict["lr/unet"] = float(lrs[-1])

            if config.train.optimizer_type.lower().startswith("dadapt"):
                log_dict["lr/d*lr"] = (
                    optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
                )

                
            ################# additional log #################
            log_dict["info/rank"] = config.network.rank
            log_dict["info/num_embeddings"] = config.network.num_embeddings
            log_dict["info/batch_size"] = config.train.batch_size
            log_dict["info/hidden_size"] = config.network.hidden_size
            log_dict["info/init_size"] = config.network.init_size
            ################# additional log #################

            log_dict.update({k: v.detach().cpu().item() for k,v in pal_k_coef_log_dict_anchor.items()})
            log_dict.update({k: v.detach().cpu().item() for k,v in pal_v_coef_log_dict_anchor.items()})
            log_dict.update({k: v.detach().cpu().item() for k,v in pal_k_coef_log_dict_erase.items()})
            log_dict.update({k: v.detach().cpu().item() for k,v in pal_v_coef_log_dict_erase.items()})
            wandb.log(log_dict, step=i)


        if len(pbar) > 10000 and i%(len(pbar)//4) == 0:
            print("Saving...")

            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}.safetensors",
                dtype=save_weight_dtype,
                metadata=model_metadata,
            )

        ####################### logging #########################     
        #########################################################
    
    return network
