import os
from transformers import T5TokenizerFast, T5EncoderModel
from diffusers import (
    PixArtSigmaPipeline
)
from diffusers.utils import (
    is_bs4_available,
    is_ftfy_available,
)
from typing import List, Sequence, Optional, Tuple, Union
import re
from tqdm import tqdm
import torch
import math
import ftfy
import html
import urllib.parse as ul
from utils.sampling import sample_in_conf_interval
from utils.image_utils import concatenate_images
from utils.parsing import find_word_token_spans
from utils.pca import pca_reconstruct, pca_reduce
from utils.merge_tmm import merge_tmm_models
from utils.tmm import TMMTorch
import argparse
import warnings

# Turn of the torch warnings
warnings.filterwarnings(action='ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def encode_prompt(
    prompt: Union[str, List[str]],
    text_encoder: T5EncoderModel,
    tokenizer: T5TokenizerFast,
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    clean_caption: bool = False,
    max_sequence_length: int = 300,
    **kwargs,
    ):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
            instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
            PixArt-Alpha, this should be "".
        do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
            whether to use classifier free guidance or not
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            number of images that should be generated per prompt
        device: (`torch.device`, *optional*):
            torch device to place the resulting embeddings on
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. For PixArt-Alpha, it's should be the embeddings of the ""
            string.
        clean_caption (`bool`, defaults to `False`):
            If `True`, the function will preprocess and clean the provided caption before encoding.
        max_sequence_length (`int`, defaults to 300): Maximum sequence length to use for the prompt.
    """

    if device is None:
        device = text_encoder.device

    # See Section 3.1. of the paper.
    max_length = max_sequence_length

    if prompt_embeds is None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to(device)

        prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)
        prompt_embeds = prompt_embeds[0]

    if text_encoder is not None:
        dtype = text_encoder.dtype
    else:
        dtype = None

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
    prompt_attention_mask = prompt_attention_mask.repeat(1, num_images_per_prompt)
    prompt_attention_mask = prompt_attention_mask.view(bs_embed * num_images_per_prompt, -1)

    return prompt_embeds, prompt_attention_mask


def stack_embeds(prompts, text_model, tokenizer, word):
    embeds = []
    spans = []

    for prompt in tqdm(prompts):
        prompt = prompt.format(word)
        span = find_word_token_spans(tokenizer, prompt, word, include_special_tokens=True, case_sensitive=False)[0]
        spans.append(span)

        with torch.no_grad():
            full_prompt_embeds, _ = encode_prompt(prompt, text_model, tokenizer, device="cuda")
            full_prompt_embeds = full_prompt_embeds.squeeze().float()
            prompt_embeds = full_prompt_embeds[span,:]
            embeds.append(prompt_embeds)
            
    embeds = torch.concatenate(embeds, dim=0)

    return embeds, spans

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduce', action='store_true')
    args = parser.parse_args()

    weight_dtype = torch.bfloat16
    text_model = T5EncoderModel.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", subfolder="text_encoder", torch_dtype=weight_dtype,).to("cuda")
    text_model = torch.compile(text_model, mode="reduce-overhead")
    tokenizer =  T5TokenizerFast.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", subfolder="tokenizer")
    checkpoint_path = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

    word = "Benedict Cumberbatch"

    with open("templates.txt", "r", encoding="utf-8") as f:
        templates = f.readlines()
    

    prompts = [prompt.rstrip("\n").format(word) for prompt in templates]
    embeds, spans = stack_embeds(prompts, text_model, tokenizer, word)
    
    embeds = embeds.to(torch.float32)
    
    if args.reduce:
        embeds, W, mean, var = pca_reduce(embeds, k=128)
        print(f"Explainable variance: {var}")

    dof_init = 2

    tmm = TMMTorch(
            n_components=8,
            covariance_type="diag",
            tied_covariance=False,
            shrinkage_alpha=0.0,
            scale_floor_scale=0.0,
            min_cluster_weight=0.0,
            learn_dof=False,    
            dof_init=dof_init,
            verbose=False,
            seed=42
        ).fit(embeds)
    
    if args.reduce:
        tmm.basis = W
        tmm.mean = mean

    prompt_gen_temp = word
    prompt_gen_embeds, prompt_attention_mask = encode_prompt(prompt_gen_temp, text_model, tokenizer, device="cuda")

    del tokenizer, text_model

    pipe = PixArtSigmaPipeline.from_pretrained(
        checkpoint_path,
        upcast_attention=False,
        torch_dtype=weight_dtype
    ).to("cuda")

    n_rand = 10
    for conf_level in range(20):

        conf_low, conf_high = 0.0+conf_level*0.05, 0.0+(conf_level+1)*0.05
        if conf_high > 0.99:
            conf_high = 0.99
        if conf_low < 0.01:
            conf_low = 0.01
        X_ring, info = sample_in_conf_interval(
            tmm,
            n=n_rand*2,
            conf_low=conf_low,
            conf_high=conf_high,
            n_ref=800_000,     
            helper_batch=65536,
            max_batches=300,
            pad_ll=0.0          
        )
        if args.reduce:
            X_ring = pca_reconstruct(X_ring, tmm.basis, tmm.mean).to(prompt_gen_embeds.dtype)
        b,d = X_ring.size()
        X_org = X_ring.detach().clone()
        X_org = X_org.reshape(b//2, 2, d)

        span = [0,1]

        save_path = "./images/t5_bennie_k128_dof2_comp32_bf"

        image = pipe(prompt_embeds=prompt_gen_embeds, prompt_attention_mask=prompt_attention_mask, generator=torch.Generator().manual_seed(0))[0][0]
        os.makedirs(f"./{save_path}/target_min{conf_low:.2f}_max{conf_high:.2f}", exist_ok=True)
        image.save(f"./{save_path}/target_min{conf_low:.2f}_max{conf_high:.2f}/org.png")

        for rand_idx in range(n_rand):
            rand_embeds = prompt_gen_embeds.clone()
            rand_embeds[:,span] = X_org[rand_idx:rand_idx+1]
            image_og = pipe(prompt_embeds=rand_embeds, prompt_attention_mask=prompt_attention_mask, generator=torch.Generator().manual_seed(0))[0][0]

            os.makedirs(f"./{save_path}/target_min{conf_low:.2f}_max{conf_high:.2f}", exist_ok=True)
            image_og.save(f"./{save_path}/target_min{conf_low:.2f}_max{conf_high:.2f}/{rand_idx}.png")

    