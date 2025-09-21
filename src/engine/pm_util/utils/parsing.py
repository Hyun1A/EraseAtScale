from typing import Sequence, List
import re
import math
from tqdm import tqdm
import torch

def _seq_find_all(seq: Sequence[int], pattern: Sequence[int]) -> List[List[int]]:
    """seq 안에서 pattern 부분수열이 등장하는 모든 시작 위치를 [연속 인덱스 리스트]로 반환"""
    hits = []
    n = len(pattern)
    if n == 0:
        return hits
    for i in range(len(seq) - n + 1):
        if seq[i:i+n] == list(pattern):
            hits.append(list(range(i, i+n)))
    return hits

def _with_special_index_map(tokenizer, prompt: str):
    """무특수 토큰 인덱스 -> 특수 토큰 포함 인덱스 매핑 테이블 생성"""
    enc_with = tokenizer(prompt, add_special_tokens=True, return_special_tokens_mask=True)
    mask = enc_with["special_tokens_mask"]
    # special_tokens_mask==0 인 위치들이 '무특수' 시퀀스의 각 위치에 대응
    no_to_with = [i for i, m in enumerate(mask) if m == 0]
    return enc_with["input_ids"], no_to_with


def find_word_token_spans(
    tokenizer,
    prompt: str,
    word: str,
    *,
    include_special_tokens: bool = False,
    case_sensitive: bool = True,
) -> List[List[int]]:

    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(
            prompt,
            add_special_tokens=include_special_tokens,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]        # (start, end) 
        ids = enc["input_ids"]

        haystack = prompt if case_sensitive else prompt.lower()
        needle = word if case_sensitive else word.lower()

        spans: List[List[int]] = []
        for m in re.finditer(re.escape(needle), haystack):
            start, end = m.span()
            token_idxs = []
            for ti, (s, e) in enumerate(offsets):
                if e <= s:
                    continue
                if not (e <= start or end <= s):
                    token_idxs.append(ti)
            if token_idxs:
                cur = [token_idxs[0]]
                for a, b in zip(token_idxs, token_idxs[1:]):
                    if b == a + 1:
                        cur.append(b)
                    else:
                        spans.append(cur)
                        cur = [b]
                spans.append(cur)
        return spans

    ids_no = tokenizer.encode(prompt, add_special_tokens=False)
    patterns = set()

    p0 = tokenizer.encode(word, add_special_tokens=False)
    if p0:
        patterns.add(tuple(p0))
    for lead in (" " , "\n", "\t"):
        p = tokenizer.encode(lead + word, add_special_tokens=False)
        if p:
            patterns.add(tuple(p))

    raw_hits_no_special: List[List[int]] = []
    seen = set()
    for pat in patterns:
        for hit in _seq_find_all(ids_no, list(pat)):
            key = tuple(hit)
            if key not in seen:
                raw_hits_no_special.append(hit)
                seen.add(key)

    if not include_special_tokens:
        return raw_hits_no_special

    _, no_to_with = _with_special_index_map(tokenizer, prompt)
    spans_with = []
    for span in raw_hits_no_special:
        spans_with.append([no_to_with[i] for i in span])
    return spans_with



def find_word_token_span_eas(
    tokenizer,
    prompt: str,
    *,
    include_special_tokens: bool = False,
    case_sensitive: bool = True,
) -> List[List[int]]:
    
    ids_no = tokenizer.encode(prompt)
    pos_eos = len(ids_no)-1

    return [[pos_eos]]





def encode_prompt(
    prompt,
    device,
    text_encoder,
    tokenizer
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        text_encoder ('CLIPTextModel'):
            text encoder - T5 is not implemented yet
        tokenizer ('CLIPTokenizerFast'):
            tokenizer - T5 is not implemented yet
    """
    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0]

    prompt_embeds_dtype = text_encoder.dtype

    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

    return prompt_embeds



def stack_embeds(prompts, text_model, tokenizer, word):
    full_embeds = []
    embeds = []
    spans = []

    for i, prompt in enumerate(tqdm(prompts)):
        prompt = prompt.format(word)
        span = find_word_token_spans(tokenizer, prompt, word, include_special_tokens=True, case_sensitive=False)[0]

        spans.append(span)

        with torch.no_grad():
            full_prompt_embeds = encode_prompt(prompt, "cuda", text_model, tokenizer).squeeze()
            prompt_embeds = full_prompt_embeds[span,:]
            full_embeds.append(full_prompt_embeds)
            embeds.append(prompt_embeds)

    full_embeds = torch.stack(full_embeds, dim=0)
    # embeds = torch.stack(embeds, dim=0)
    embeds = torch.concatenate(embeds, dim=0)
    # embeds = embeds.reshape(embeds.shape[0], -1)

    return full_embeds, embeds, spans



def stack_embeds_last_token(prompts, text_model, tokenizer, word):
    full_embeds = []
    embeds = []
    spans = []


    len_subset = 512
    full_prompt_embeds_stack = []
    for i in range(math.ceil(len(prompts)/float(len_subset))):
        if (i+1)*len_subset > len(prompts):
            prompts_shard = prompts[i*len_subset:]
        else:
            prompts_shard = prompts[i*len_subset:(i+1)*len_subset]

        full_prompt_embeds_shard = encode_prompt(prompts_shard, "cuda", text_model, tokenizer)
        full_prompt_embeds_stack.append(full_prompt_embeds_shard)

    full_prompt_embeds_stack = torch.cat(full_prompt_embeds_stack, dim=0)


    for i, prompt in enumerate(tqdm(prompts)):
        span = find_word_token_spans(tokenizer, prompt, word, include_special_tokens=True, case_sensitive=False)[0]

        span = span[-1:]
        spans.append(span)

        with torch.no_grad():
            full_prompt_embeds = full_prompt_embeds_stack[i]
            prompt_embeds = full_prompt_embeds[span,:]
            full_embeds.append(full_prompt_embeds)
            embeds.append(prompt_embeds)


    full_embeds = torch.stack(full_embeds, dim=0)
    embeds = torch.concatenate(embeds, dim=0)


    return full_embeds, embeds, spans



def stack_embeds_eos(prompts, text_model, tokenizer):
    full_embeds = []
    embeds = []
    spans = []


    len_subset = 512
    full_prompt_embeds_stack = []
    for i in range(math.ceil(len(prompts)/float(len_subset))):
        if (i+1)*len_subset > len(prompts):
            prompts_shard = prompts[i*len_subset:]
        else:
            prompts_shard = prompts[i*len_subset:(i+1)*len_subset]

        full_prompt_embeds_shard = encode_prompt(prompts_shard, "cuda", text_model, tokenizer)
        full_prompt_embeds_stack.append(full_prompt_embeds_shard)

    full_prompt_embeds_stack = torch.cat(full_prompt_embeds_stack, dim=0)


    for i, prompt in enumerate(tqdm(prompts)):
        span = find_word_token_span_eas(tokenizer, prompt, include_special_tokens=True, case_sensitive=False)[0]

        span = span[-1:]
        spans.append(span)

        with torch.no_grad():
            full_prompt_embeds = full_prompt_embeds_stack[i]
            prompt_embeds = full_prompt_embeds[span,:]
            full_embeds.append(full_prompt_embeds)
            embeds.append(prompt_embeds)

    full_embeds = torch.stack(full_embeds, dim=0)
    embeds = torch.concatenate(embeds, dim=0)


    return full_embeds, embeds, spans




def stack_embeds_none(prompts, text_model, tokenizer, word):
    full_embeds = []
    embeds = []

    len_subset = 512
    full_prompt_embeds_stack = []
    for i in range(math.ceil(len(prompts)/float(len_subset))):
        if (i+1)*len_subset > len(prompts):
            prompts_shard = prompts[i*len_subset:]
        else:
            prompts_shard = prompts[i*len_subset:(i+1)*len_subset]

        full_prompt_embeds_shard = encode_prompt(prompts_shard, "cuda", text_model, tokenizer)
        full_prompt_embeds_stack.append(full_prompt_embeds_shard)

    full_prompt_embeds_stack = torch.cat(full_prompt_embeds_stack, dim=0)


    return full_embeds, embeds