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
from utils.transport import optimal_transport
from utils.tmm import TMMTorch
import argparse
import warnings

# Turn of the torch warnings
warnings.filterwarnings(action='ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if is_bs4_available():
    from bs4 import BeautifulSoup

# Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption
def _clean_caption(caption):
    caption = str(caption)
    caption = ul.unquote_plus(caption)
    caption = caption.strip().lower()
    caption = re.sub("<person>", "person", caption)
    # urls:
    caption = re.sub(
        r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
        "",
        caption,
    )  # regex for urls
    caption = re.sub(
        r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
        "",
        caption,
    )  # regex for urls
    # html:
    caption = BeautifulSoup(caption, features="html.parser").text

    # @<nickname>
    caption = re.sub(r"@[\w\d]+\b", "", caption)

    # 31C0—31EF CJK Strokes
    # 31F0—31FF Katakana Phonetic Extensions
    # 3200—32FF Enclosed CJK Letters and Months
    # 3300—33FF CJK Compatibility
    # 3400—4DBF CJK Unified Ideographs Extension A
    # 4DC0—4DFF Yijing Hexagram Symbols
    # 4E00—9FFF CJK Unified Ideographs
    caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
    caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
    caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
    caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
    caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
    caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
    caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
    #######################################################

    # все виды тире / all types of dash --> "-"
    caption = re.sub(
        r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
        "-",
        caption,
    )

    # кавычки к одному стандарту
    caption = re.sub(r"[`´«»“”¨]", '"', caption)
    caption = re.sub(r"[‘’]", "'", caption)

    # &quot;
    caption = re.sub(r"&quot;?", "", caption)
    # &amp
    caption = re.sub(r"&amp", "", caption)

    # ip adresses:
    caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

    # article ids:
    caption = re.sub(r"\d:\d\d\s+$", "", caption)

    # \n
    caption = re.sub(r"\\n", " ", caption)

    # "#123"
    caption = re.sub(r"#\d{1,3}\b", "", caption)
    # "#12345.."
    caption = re.sub(r"#\d{5,}\b", "", caption)
    # "123456.."
    caption = re.sub(r"\b\d{6,}\b", "", caption)
    # filenames:
    caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

    #
    caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
    caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""
    caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

    # this-is-my-cute-cat / this_is_my_cute_cat
    regex2 = re.compile(r"(?:\-|\_)")
    if len(re.findall(regex2, caption)) > 3:
        caption = re.sub(regex2, " ", caption)

    caption = ftfy.fix_text(caption)
    caption = html.unescape(html.unescape(caption))

    caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
    caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
    caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

    caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
    caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
    caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
    caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
    caption = re.sub(r"\bpage\s+\d+\b", "", caption)

    caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

    caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

    caption = re.sub(r"\b\s+\:\s+", r": ", caption)
    caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
    caption = re.sub(r"\s+", " ", caption)

    caption.strip()

    caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
    caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
    caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
    caption = re.sub(r"^\.\S+$", "", caption)

    return caption.strip()

def _text_preprocessing(text, clean_caption=False):
    if clean_caption and not is_bs4_available():
        clean_caption = False

    if clean_caption and not is_ftfy_available():
        clean_caption = False

    if not isinstance(text, (tuple, list)):
        text = [text]

    def process(text: str):
        if clean_caption:
            text = _clean_caption(text)
            text = _clean_caption(text)
        else:
            text = text.lower().strip()
        return text

    return [process(t) for t in text]

@torch.no_grad()
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
        # prompt = _text_preprocessing(prompt, clean_caption=clean_caption)
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
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduce', action='store_true')
    args = parser.parse_args()

    weight_dtype = torch.bfloat16
    text_model = T5EncoderModel.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", subfolder="text_encoder", torch_dtype=weight_dtype,).to("cuda")
    text_model = torch.compile(text_model, mode="reduce-overhead")
    tokenizer =  T5TokenizerFast.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", subfolder="tokenizer")
    checkpoint_path = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

    word = "Brad Pitt"

    with open("templates.txt", "r", encoding="utf-8") as f:
        templates = f.readlines()
    

    prompts = [prompt.rstrip("\n").format(word) for prompt in templates]
    embeds, spans = stack_embeds(prompts, text_model, tokenizer, word)
    embeds = embeds.to(torch.float32)

    if args.reduce:
        embeds, W, mean, var = pca_reduce(embeds, k=16)
        print(f"Explainable variance: {var}")

    dof_init = 2.0

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

    tgt_words = ["Benedict Cumberbatch"]
    tgt_tmms = []

    for word in tgt_words:
        prompts = [prompt.rstrip("\n").format(word) for prompt in templates]
        embeds, spans = stack_embeds(prompts, text_model, tokenizer, word)
        embeds = embeds.to(torch.float32)
        if args.reduce:
            embeds, W, mean, var = pca_reduce(embeds, k=16)
            print(f"Explainable variance: {var}")

        new_tmm = TMMTorch(
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
        
        new_tmm.basis = W
        new_tmm.mean = mean
        tgt_tmms.append(new_tmm)

    tmm_tgt = merge_tmm_models(tgt_tmms)
    prompt_gen_temp = "Brad Pitt"
    prompt_gen_embeds, prompt_attention_mask = encode_prompt(prompt_gen_temp, text_model, tokenizer, device="cuda")
    original_token_len = prompt_attention_mask.sum().item() - 1
    # replace eos token
    # with torch.no_grad():
    #     prompt_embeds_eos, prompt_attention_mask_eos = encode_prompt(tgt_words[0], text_model, tokenizer, device="cuda")
    #     eos_token = prompt_embeds_eos[0, prompt_attention_mask_eos.sum().item() - 1]
    #     prompt_gen_embeds[0, prompt_attention_mask.sum().item() - 1] = eos_token

    del tokenizer, text_model

    pipe = PixArtSigmaPipeline.from_pretrained(
        checkpoint_path,
        upcast_attention=False,
        torch_dtype=weight_dtype
    ).to("cuda")

    n_rand = 10
    for conf_level in range(19):

        conf_low, conf_high = 0.05+conf_level*0.05, 0.05+(conf_level+1)*0.05
        if conf_low < 0.01:
            conf_low = 0.01
        if conf_high > 0.99:
            conf_high = 0.99
        X_ring, info = sample_in_conf_interval(
            tmm,
            n=n_rand * original_token_len,
            conf_low=conf_low,
            conf_high=conf_high,
            n_ref=800_000,     
            helper_batch=65536,
            max_batches=300,
            pad_ll=0.0          
        )
        if args.reduce:
            X_ring_recon = pca_reconstruct(X_ring, tmm.basis, tmm.mean).to(prompt_gen_embeds.dtype)
        b,d = X_ring_recon.size()
        X_org = X_ring_recon.detach().clone()
        X_org = X_org.reshape(b//original_token_len, original_token_len, d)

        X_ring, _ = optimal_transport(tmm, tmm_tgt, X_ring, return_plan=True)
        if args.reduce:
            X_ring = pca_reconstruct(X_ring, tmm_tgt.basis, tmm_tgt.mean)
        X_ring = X_ring.reshape(b//original_token_len, original_token_len, d).to(prompt_gen_embeds.dtype)

        span = list(range(original_token_len))

        save_path = "./images/t5_leonardo_to_many_bf_k16_c8"

        image = pipe(prompt_embeds=prompt_gen_embeds, prompt_attention_mask=prompt_attention_mask, generator=torch.Generator().manual_seed(0))[0][0]
        os.makedirs(f"./{save_path}/target_min{conf_low:.2f}_max{conf_high:.2f}", exist_ok=True)
        image.save(f"./{save_path}/target_min{conf_low:.2f}_max{conf_high:.2f}/org.png")

        for rand_idx in range(n_rand):
            rand_embeds = prompt_gen_embeds.clone()
            rand_embeds[:,span] = X_ring[rand_idx:rand_idx+1]
            image_ot = pipe(prompt_embeds=rand_embeds, prompt_attention_mask=prompt_attention_mask, generator=torch.Generator().manual_seed(0))[0][0]

            rand_embeds = prompt_gen_embeds.clone()
            rand_embeds[:,span] = X_org[rand_idx:rand_idx+1]
            image_og = pipe(prompt_embeds=rand_embeds, prompt_attention_mask=prompt_attention_mask, generator=torch.Generator().manual_seed(0))[0][0]

            image = concatenate_images(image_og, image_ot)

            os.makedirs(f"./{save_path}/target_min{conf_low:.2f}_max{conf_high:.2f}", exist_ok=True)
            image.save(f"./{save_path}/target_min{conf_low:.2f}_max{conf_high:.2f}/{rand_idx}.png")

    