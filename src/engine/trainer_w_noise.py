from pathlib import Path
import gc
from copy import deepcopy
import pandas as pd
import random

import torch
from tqdm import tqdm
import os, sys
import numpy as np

sys.path[0] = "/".join(sys.path[0].split('/')[:-1])

import wandb

def train_erase_one_stage(
            stage,
            pbar,
            config,
            device_cuda,
            pipe,
            unet,
            tokenizer,
            text_encoder,
            network,
            network_modules,
            unet_modules,
            optimizer,
            lr_scheduler,
            criteria,
            prompts,
            replace_word,
            embedding_unconditional,
            anchor_sampler,
            lipschitz,
            save_weight_dtype,
            model_metadata,
            args,
            noise_dict,
            data_loader,
            prompt_one,
            network_teacher,
            save_path,
            ):

    data_iter = iter(data_loader)

    for i in pbar:
        loss=dict()
        optimizer.zero_grad()




        
        #####################################################################
        ################### Prepare training embeddings #####################

        targets, mapping, anchors = next(data_iter)  # CPU tensors
        targets = targets.to(device_cuda, non_blocking=True).float()
        mapping = mapping.to(device_cuda, non_blocking=True).float()
        anchors = anchors.to(device_cuda, non_blocking=True).float()


        ################### Prepare training embeddings #####################
        #####################################################################        
        
        

        #########################################################
        ############### loss_prompt_erase/anchor ################   
        pal = torch.tensor([config.train.pal]).float().to(device=device_cuda)     

        pal_k_coef_log_dict_erase = dict()
        pal_v_coef_log_dict_erase = dict()        
        loss_prompt_erase_to_k = 0
        loss_prompt_erase_to_v = 0

        pal_k_coef_log_dict_anchor = dict()
        pal_v_coef_log_dict_anchor = dict()
        loss_prompt_anchor_to_k = 0
        loss_prompt_anchor_to_v = 0   

        loss_adv_erase_to_k = torch.tensor([0.]).float().to(device=device_cuda)     
        loss_adv_erase_to_v = torch.tensor([0.]).float().to(device=device_cuda)     
        
        idx = 0
        uncond = embedding_unconditional[:, :2]
    
        for name in network_modules.keys():
            if not "lora_unet" in name:
                continue
            if "lora_adaptor" in name:
                continue


            with torch.no_grad():
                crsattn_org_noeas = unet_modules[name](torch.cat([targets, mapping, uncond, anchors], dim=0).float())
                crsattn_target_org_noeas = crsattn_org_noeas[0].unsqueeze(0) #if targets.size(0) == 1 else crsattn_org[:1]
                crsattn_neutral_org_noeas = crsattn_org_noeas[1].unsqueeze(0)
                crsattn_comp_org_noeas = crsattn_org_noeas[2:]

                with network_teacher:
                    crsattn_org = unet_modules[name](torch.cat([targets, mapping, uncond, anchors], dim=0).float())
                    crsattn_target_org = crsattn_org[0].unsqueeze(0) #if targets.size(0) == 1 else crsattn_org[:1]
                    crsattn_neutral_org = crsattn_org[1].unsqueeze(0)
                    crsattn_comp_org = crsattn_org[2:]

                
            weight = unet_modules[name].weight.data
            weight_noise = noise_dict[name]
            unet_modules[name].weight.data = (weight + weight_noise).clone()


            with network:
                crsattn = unet_modules[name](torch.cat([targets, mapping, uncond, anchors], dim=0).float())
                crsattn_target = crsattn[0].unsqueeze(0) #if targets.size(0) == 1 else crsattn[:1]
                crsattn_comp = crsattn[2:]

                
            g_scale = prompt_one[0].guidance_scale
            if "to_k" in name:
                lipschitz_for_key_target = ( lipschitz['lipschitz_ov'][idx]*lipschitz['lipschitz_q'][idx] ).unsqueeze(0).unsqueeze(1).unsqueeze(2) / 77
                loss_prompt_erase_to_k += (lipschitz_for_key_target * (crsattn_target_org - crsattn_target)**2).mean()
                pal_k_coef_log_dict_erase[f"pal_k_coef_log_dict_erase/{idx}th-layer"] = lipschitz_for_key_target.mean()

                lipschitz_for_key_comp = ( lipschitz['lipschitz_ov'][idx]*lipschitz['lipschitz_q'][idx] ).unsqueeze(0).unsqueeze(1).unsqueeze(2) / 77
                loss_prompt_anchor_to_k += (lipschitz_for_key_comp * (crsattn_comp_org-crsattn_comp)**2).mean()
                pal_k_coef_log_dict_anchor[f"pal_k_coef_log_dict_anchor/{idx}th-layer"] = lipschitz_for_key_comp.mean()

            else:
                lipschitz_for_val_target = lipschitz['lipschitz_o'][idx].unsqueeze(0).unsqueeze(1)
                loss_prompt_erase_to_v += (lipschitz_for_val_target * ( (crsattn_neutral_org - crsattn_target) )**2).mean()
                pal_v_coef_log_dict_erase[f"pal_v_coef_log_dict_erase/{idx}th-layer"] = lipschitz_for_val_target.mean()

                lipschitz_for_val_comp = lipschitz['lipschitz_o'][idx].unsqueeze(0).repeat(crsattn_comp.shape[0],1).unsqueeze(2)
                loss_prompt_anchor_to_v += (lipschitz_for_val_comp * (crsattn_comp_org-crsattn_comp)**2).mean() #/ crsattn_comp_org.shape[0]
                pal_v_coef_log_dict_anchor[f"pal_v_coef_log_dict_anchor/{idx}th-layer"] = lipschitz_for_val_comp.mean()                

                idx+=1
        
        
        
        loss_prompt_erase_to_k = loss_prompt_erase_to_k / len(network_modules)
        loss_prompt_erase_to_v = loss_prompt_erase_to_v / len(network_modules)
        loss_prompt_erase = loss_prompt_erase_to_v + loss_prompt_erase_to_k       

        loss_prompt_anchor_to_k = loss_prompt_anchor_to_k / len(network_modules)
        loss_prompt_anchor_to_v = loss_prompt_anchor_to_v / len(network_modules)
        loss_prompt_anchor = loss_prompt_anchor_to_v + loss_prompt_anchor_to_k        


        loss_adv_erase_to_k = loss_adv_erase_to_k / len(network_modules)
        loss_adv_erase_to_v = loss_adv_erase_to_v / len(network_modules)
        loss_adv_erase = loss_adv_erase_to_v + loss_adv_erase_to_k


        
        loss[f"loss_erasing_stage{stage}/loss_prompt_erase"] = loss_prompt_erase
        loss[f"loss_erasing_stage{stage}/loss_prompt_erase_to_k"] = loss_prompt_erase_to_k
        loss[f"loss_erasing_stage{stage}/loss_prompt_erase_to_v"] = loss_prompt_erase_to_v

        loss[f"loss_erasing_stage{stage}/loss_prompt_anchor"] = loss_prompt_anchor
        loss[f"loss_erasing_stage{stage}/loss_prompt_anchor_to_k"] = loss_prompt_anchor_to_k
        loss[f"loss_erasing_stage{stage}/loss_prompt_anchor_to_v"] = loss_prompt_anchor_to_v

        loss[f"loss_erasing_stage{stage}/loss_adv_erase"] = loss_adv_erase 
        
        adv_coef = config.train.adv_coef
        loss[f"loss_erasing"] = loss[f"loss_erasing_stage{stage}/loss_prompt_erase"] \
                            + adv_coef * loss[f"loss_erasing_stage{stage}/loss_adv_erase"] \
                            + pal * loss[f"loss_erasing_stage{stage}/loss_prompt_anchor"]



        ############### loss_prompt_erase/anchor ################        
        #########################################################                


        #########################################################        
        ######################### misc ##########################    
        loss["pal"] = pal
        loss["guidance"] = torch.tensor([prompt_one[0].guidance_scale]).cuda()
        loss["la_strength"] = torch.tensor([prompt_one[0].la_strength]).cuda()
        loss["batch_size"] = torch.tensor([prompt_one[0].batch_size]).cuda()
        ######################### misc ##########################        
        #########################################################        

        #########################################################        
        ######################## optim ##########################        
        loss[f"loss_erasing"].backward()


        optimizer.step()
        lr_scheduler.step()

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
            log_dict["info/adv_coef"] = adv_coef
            ################# additional log #################

            log_dict.update({k: v.detach().cpu().item() for k,v in pal_k_coef_log_dict_anchor.items()})
            log_dict.update({k: v.detach().cpu().item() for k,v in pal_v_coef_log_dict_anchor.items()})
            log_dict.update({k: v.detach().cpu().item() for k,v in pal_k_coef_log_dict_erase.items()})
            log_dict.update({k: v.detach().cpu().item() for k,v in pal_v_coef_log_dict_erase.items()})
            wandb.log(log_dict, step=i)


        if len(pbar) > 50000 and i%(len(pbar)//10) == 0:
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
