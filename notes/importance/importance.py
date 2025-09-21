import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

domain = "character"
load_path = f"/home/ldlqudgus756/EraseAtScale/EAS/EAS_sd1.4/importance_buffer/hutchinson_est/{domain}"

con_list = os.listdir(load_path)

importance_t = dict()
importance_a = dict()

for con in con_list:
    try:
        importance_t_samp = torch.load(f"{load_path}/{con}/stacked_target_importance.pt")
        importance_a_samp = torch.load(f"{load_path}/{con}/stacked_anchor_importance.pt")
    except:
        continue

    key_t = next(iter(importance_t_samp.keys()))
    key_a = next(iter(importance_a_samp.keys()))

    # print(importance_t_samp[key_t]['up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v'].max())
    # print(importance_a_samp[key_a]['up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v'].max())

    for key in importance_t_samp[key_t].keys():
        if importance_t.get(key) is not None:
            importance_t[key] += importance_t_samp[key_t][key]
            importance_a[key] += importance_a_samp[key_a][key]
        else:
            importance_t[key] = importance_t_samp[key_t][key]
            importance_a[key] = importance_a_samp[key_a][key]
        


rel_importance = dict()

for key in importance_t.keys():
    rel_importance[key] = importance_t[key]  #- importance_a[key] #, min=0)
    # rel_importance[key] = importance_t[key] / (importance_a[key]+1e-15)


ca_var_k = []
ca_var_v = []
a_name = []

for k,v in rel_importance.items():
    if "to_k" in k:
        ca_var_k.append(v.max().item())
        a_name.append(k[:-4])
    elif "to_v" in k:
        ca_var_v.append(v.max().item())

ca_var_k = np.array(ca_var_k)
ca_var_v = np.array(ca_var_v)

plt.figure(figsize=(20,10))

plt.plot(range(len(a_name)), ca_var_k)
plt.plot(range(len(a_name)), ca_var_v)


plt.legend(["key", "value"])


plt.xticks(range(len(a_name)), a_name, rotation=90)
plt.tight_layout()

plt.savefig(f"./{domain}_samp.png")

