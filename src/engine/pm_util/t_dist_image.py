import os
from transformers import CLIPTextModel, CLIPTokenizerFast
from diffusers import (
    StableDiffusionPipeline,
)
from diffusers import LMSDiscreteScheduler
import torch
import torch
from PIL import Image
import argparse
from utils.tmm import TMMTorch
from utils.image_utils import concatenate_images
from utils.pca import pca_reconstruct, pca_reduce
from utils.parsing import stack_embeds, encode_prompt
from utils.sampling import sample_in_conf_interval
from utils.merge_tmm import merge_tmm_models
from utils.transport import optimal_transport, tmm_emd
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@torch.no_grad()
def embedding2img(embeddings, pipe, device='cuda:0', guidance_scale=7.5, image_size=512, precision=torch.float16, \
                  ddim_steps=50, seed=0, uncond_embeddings=None):
        
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
        
    height = image_size
    width = image_size                         
    num_inference_steps = ddim_steps           # Number of denoising steps

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

    text_embeddings = torch.cat([uncond_embeddings, embeddings]).to(precision)
    
    latents = torch.randn(
        (1, unet.config.in_channels, height // 8, width // 8),
        generator=generator,device=torch_device
    ).repeat(batch_size,1,1,1).to(precision)

    latents = latents.to(torch_device)

    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.init_noise_sigma

    from tqdm.auto import tqdm

    scheduler.set_timesteps(num_inference_steps)
 
    for idx, t in enumerate(tqdm(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample


        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")

    return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduce', action='store_true')
    args = parser.parse_args()

    text_model = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder", variant='fp16').to("cuda")
    tokenizer =  CLIPTokenizerFast.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")


    checkpoint_path = "CompVis/stable-diffusion-v1-4"

    DIFFUSERS_CACHE_DIR = ".cache/"  # if you want to change the cache dir, change this
    LOCAL_ONLY = False  # if you want to use only local files, change this

    pipe = StableDiffusionPipeline.from_pretrained(
        checkpoint_path,
        upcast_attention=False,
        torch_dtype=torch.float16,
        cache_dir=DIFFUSERS_CACHE_DIR,
        local_files_only=LOCAL_ONLY,
    ).to("cuda")

    word = "Chris Evans"

    with open("templates.txt", "r", encoding="utf-8") as f:
        templates = f.readlines()
    

    prompts = [prompt.rstrip("\n").format(word) for prompt in templates]
    full_embeds, embeds, spans = stack_embeds(prompts, text_model, tokenizer, word)
    embeds_clone = embeds.clone()
    if args.reduce:
        embeds, W, mean, var = pca_reduce(embeds, k=32)
        print(f"Explainable variance: {var}")
    dof_init = 2

    tmm = TMMTorch(
            n_components=4,
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
    
    # tgt_words = ["Leonardo Dicaprio", "Chris Evans", "Chris Hemsworth", "Emma Stone"]
    tgt_words = ["Chris Hemsworth"]
    tgt_tmms = []

    for word in tgt_words:
        prompts = [prompt.rstrip("\n").format(word) for prompt in templates]
        full_embeds, embeds, spans = stack_embeds(prompts, text_model, tokenizer, word)
        if args.reduce:
            embeds, W, mean, var = pca_reduce(embeds, k=32)
            print(f"Explainable variance: {var}")

        new_tmm = TMMTorch(
                n_components=4,
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
        
        new_tmm.basis = W
        new_tmm.mean = mean
        tgt_tmms.append(new_tmm)

            

    tmm_tgt = merge_tmm_models(tgt_tmms)

    prompt_gen_temp = "Chris Evans"
    prompt_gen_embeds = encode_prompt(prompt_gen_temp, "cuda", text_model, tokenizer)

    n_rand = 5
    for conf_level in range(20):
        conf_low, conf_high = 0.05+conf_level*0.05, 0.05+(conf_level+1)*0.05
        if conf_high > 0.99:
            conf_high = 0.99
        X_ring, info = sample_in_conf_interval(
            tmm,
            n=n_rand,
            conf_low=conf_low,
            conf_high=conf_high,
            n_ref=800_000,     
            helper_batch=65536,
            max_batches=300,
            pad_ll=0.0          
        )
        b,d = X_ring.size()
        if args.reduce:
            X_ring_recon = pca_reconstruct(X_ring, tmm.basis, tmm.mean)
        X_org = X_ring_recon.detach().clone()
        X_org = X_org.reshape(b, 2, 768)

        X_ring, _ = optimal_transport(tmm, tmm_tgt, X_ring, return_plan=True)
        if args.reduce:
            X_ring = pca_reconstruct(X_ring, tmm_tgt.basis, tmm_tgt.mean)
        X_ring = X_ring.reshape(b, 2, 768)

        span = [1,2]

        save_path = "./images/ce_to_ch_stack"

        prompt_gen_embeds = prompt_gen_embeds[:,:4]
        images = embedding2img(prompt_gen_embeds, pipe, seed=42)[0]
        image = Image.fromarray(images)
        os.makedirs(f"./{save_path}/target_min{conf_low:.2f}_max{conf_high:.2f}", exist_ok=True)
        image.save(f"./{save_path}/target_min{conf_low:.2f}_max{conf_high:.2f}/org.png")

        for rand_idx in range(n_rand):
            rand_embeds = prompt_gen_embeds.clone()
            rand_embeds[:,span] = X_ring[rand_idx:rand_idx+1]
            images = embedding2img(rand_embeds, pipe, seed=42)[0]
            image_ot = Image.fromarray(images)

            rand_embeds = prompt_gen_embeds.clone()
            rand_embeds[:,span] = X_org[rand_idx:rand_idx+1]
            images = embedding2img(rand_embeds, pipe, seed=42)[0]
            image_og = Image.fromarray(images)

            image = concatenate_images(image_og, image_ot)

            os.makedirs(f"./{save_path}/target_min{conf_low:.2f}_max{conf_high:.2f}", exist_ok=True)
            image.save(f"./{save_path}/target_min{conf_low:.2f}_max{conf_high:.2f}/{rand_idx}.png")

    