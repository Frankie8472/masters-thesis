import gc
import json
import os
import sys

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, \
    set_seed


def create_corpus(
        tokenizer_name="gpt2",
        model_name="gpt2",
        max_document_length=None,
        device="cpu",
        corpus_size=1,
        tokenizer_model=AutoTokenizer,
        lm_model=AutoModelForCausalLM,
        pad_token_id=None,
        save_path="data/test",
        load_size=1
    ):
    r"""
    Generates sequences/documents/a corpus for models with a language modeling head.

    Parameters:
        corpus_size (`int`, *optional*, defaults to 1):
            The corpus size to be generated (number of documents)
        model_name (`str`, *optional*, defaults to "openai-gpt"):
            The model name of the pre-trained model: openai-gpt, gpt2-small, gpt2, gpt2-large, gpt2-xl, transfo-xl-wt103, EleutherAI/gpt-neo-2.7B, ctrl
        max_document_length (`int`, *optional*, defaults to None):
            The maximum document length, normally set to tokenizer.max_length
        tokenizer_model (`PreTrainedTokenizer`, *optional*, defaults to AutoTokenizer):
            The pre-trained tokenizer class
        lm_model (`PreTrainedModel`, *optional*, defaults to AutoModelForCausalLM):
            The pre-trained model class with language modeling head
        device (`str`, *optional*, defaults to "cpu"):
            The device the computations commence "cpu" or "cuda"
    """

    tokenizer = tokenizer_model.from_pretrained(tokenizer_name)
    model = lm_model.from_pretrained(model_name)

    max_document_length = max_document_length if max_document_length is not None else tokenizer.model_max_length
    if pad_token_id is not None:
        if pad_token_id == 'eos_token_id':
            pad_token_id = tokenizer.eos_token_id
        else:
            print("ERROR: Undefinded/unimplemented pad_token_id")

    model = model.to(device)

    decoded_output = []

    for i in tqdm(range(0, corpus_size, load_size)):
        step_size = min(load_size, corpus_size-i)
        encoded_output = model.generate(
            # all parameters have to be set as otherwise the config of the pretrained model will be taken
            input_ids=None,
            max_length=max_document_length,
            do_sample=True,                         # False implies Greedy search
            early_stopping=False,
            num_beams=1,                            # 1 deactivates beam_search
            temperature=1.0,                        # 1.0 deactivates temperature
            top_k=0,                                # 0 deactivates top_k sampling
            top_p=1.0,                              # 1.0 deactivates top_p sampling
            repetition_penalty=1.0,                 # 1.0 deactivates repetition_penalty
            pad_token_id=pad_token_id,              # For open-end generation set to eos_token_id
            #bos_token_id=bos_token_id,
            #eos_token_id=eos_token_id,
            length_penalty=1.0,                     # 1.0 deactivates length_penalty
            no_repeat_ngram_size=0,                 # 0 deactivates no_repeat_ngram_size
            encoder_no_repeat_ngram_size=0,         # 0 deactivates encoder_no_repeat_ngram_size
            num_return_sequences=step_size,       # The number of independently computed returned sequences for each element in the batch. No input means batch size of one.
            num_beam_groups=1,
            output_scores=False,                    # Will be important if you want the prediction scores!
        )

        for j in range(step_size):
            decoded_output.append(tokenizer.decode(encoded_output[j], skip_special_tokens=True))

    print(decoded_output)

    gc.collect()
    torch.cuda.empty_cache()


def get_metadata():
    with open('data/data_arxiv-metadata-oai-snapshot.json', 'r') as f:
        for line in f:
            yield line


def main():
    o = 0
    metadata = get_metadata()
    for paper in metadata:
        #print(json.loads(paper)['abstract'])
        o += 1

    print(o)

main()
