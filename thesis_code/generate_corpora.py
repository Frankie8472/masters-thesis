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

    if os.path.isfile(save_path):
        print("ERROR: file already exist, please remove manually before running again.")
        return

    tokenizer = tokenizer_model.from_pretrained(tokenizer_name)
    model = lm_model.from_pretrained(model_name)

    max_document_length = max_document_length if max_document_length is not None else tokenizer.model_max_length
    if pad_token_id is not None:
        if pad_token_id == 'eos_token_id':
            pad_token_id = tokenizer.eos_token_id
        else:
            print("ERROR: Undefinded/unimplemented pad_token_id")

    # print(f"EOS: {tokenizer.eos_token} | BOS: {tokenizer.bos_token} | UNK: {tokenizer.unk_token}")

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
            top_p=1.0,                              # 1.0 deactivates top_p (nucleus) sampling  using 0.9
            typical_p=0.2,                          # 1.0 deactivates typical_p sampling        using 0.2
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

    with open(save_path, 'w') as file:
        json.dump(decoded_output, file, indent=2)

    gc.collect()
    torch.cuda.empty_cache()


def main():
    corpus_size = 100000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_size = 100

    if len(sys.argv) < 2:
        print("ERROR: No input argument")
        return

    case = int(sys.argv[1])

    set_seed(42)

    # Integrated
    if case == 1:
        print(f"Entering case: {case}")
        create_corpus(
            tokenizer_name="gpt2",
            model_name="/cluster/work/cotterell/knobelf/data/model-gpt2-wiki-integrated",
            max_document_length=None,
            device=device,
            corpus_size=corpus_size,
            tokenizer_model=GPT2Tokenizer,
            lm_model=GPT2LMHeadModel,
            pad_token_id='eos_token_id',
            save_path="/cluster/work/cotterell/knobelf/data/dataset1-gpt2-wiki-integrated.json",
            load_size=load_size
        )

    # Self
    elif case == 2:
        print(f"Entering case: {case}")
        create_corpus(
            tokenizer_name="gpt2",
            model_name="/cluster/work/cotterell/knobelf/data/model-gpt2-wiki",
            max_document_length=None,
            device=device,
            corpus_size=corpus_size,
            tokenizer_model=GPT2Tokenizer,
            lm_model=GPT2LMHeadModel,
            pad_token_id='eos_token_id',
            save_path="/cluster/work/cotterell/knobelf/data/dataset1-gpt2-wiki.json",
            load_size=load_size
        )

    # No titles
    elif case == 3:
        print(f"Entering case: {case}")
        create_corpus(
            tokenizer_name="gpt2",
            model_name="/cluster/work/cotterell/knobelf/data/model-gpt2-wiki_nt",
            max_document_length=None,
            device=device,
            corpus_size=corpus_size,
            tokenizer_model=GPT2Tokenizer,
            lm_model=GPT2LMHeadModel,
            pad_token_id='eos_token_id',
            save_path="/cluster/work/cotterell/knobelf/data/dataset1-gpt2-wiki_nt-typ_p.json",
            load_size=load_size
        )
    elif case == 4:
        print(f"Entering case: {case}")
        set_seed(1337)
        create_corpus(
            tokenizer_name="gpt2",
            model_name="/cluster/work/cotterell/knobelf/data/model-gpt2-wiki_nt",
            max_document_length=None,
            device=device,
            corpus_size=corpus_size,
            tokenizer_model=GPT2Tokenizer,
            lm_model=GPT2LMHeadModel,
            pad_token_id='eos_token_id',
            save_path="/cluster/work/cotterell/knobelf/data/dataset2-gpt2-wiki_nt-typ_p.json",
            load_size=load_size
        )
    elif case == 5:
        print(f"Entering case: {case}")
        create_corpus(
            tokenizer_name="gpt2",
            model_name="gpt2",
            max_document_length=None,
            device=device,
            corpus_size=corpus_size,
            tokenizer_model=GPT2Tokenizer,
            lm_model=GPT2LMHeadModel,
            pad_token_id='eos_token_id',
            save_path="/cluster/work/cotterell/knobelf/data/dataset1-gpt2.json",
            load_size=load_size
        )
    elif case == 6:
        print(f"Entering case: {case}")
        set_seed(1337)
        create_corpus(
            tokenizer_name="gpt2",
            model_name="gpt2",
            max_document_length=None,
            device=device,
            corpus_size=corpus_size,
            tokenizer_model=GPT2Tokenizer,
            lm_model=GPT2LMHeadModel,
            pad_token_id='eos_token_id',
            save_path="/cluster/work/cotterell/knobelf/data/dataset2-gpt2.json",
            load_size=load_size
        )
    elif case == 7:
        print(f"Entering case: {case}")
        create_corpus(
            tokenizer_name="gpt2-large",
            model_name="gpt2-large",
            max_document_length=None,
            device=device,
            corpus_size=corpus_size,
            tokenizer_model=GPT2Tokenizer,
            lm_model=GPT2LMHeadModel,
            pad_token_id='eos_token_id',
            save_path="/cluster/work/cotterell/knobelf/data/data_gpt2-large.json",
            load_size=load_size
        )
    elif case == 8:
        print(f"Entering case: {case}")
        create_corpus(
            tokenizer_name="gpt2-xl",
            model_name="gpt2-xl",
            max_document_length=None,
            device=device,
            corpus_size=corpus_size,
            tokenizer_model=GPT2Tokenizer,
            lm_model=GPT2LMHeadModel,
            pad_token_id='eos_token_id',
            save_path="/cluster/work/cotterell/knobelf/data/data_gpt2-xl.json",
            load_size=load_size
        )
    elif case == 9:
        print(f"Entering case: {case}")
        create_corpus(
            tokenizer_name="EleutherAI/gpt-neo-2.7B",
            model_name="EleutherAI/gpt-neo-2.7B",
            max_document_length=2048,
            device=device,
            corpus_size=corpus_size,
            tokenizer_model=GPT2Tokenizer,
            lm_model=GPTNeoForCausalLM,
            pad_token_id=None,
            save_path="/cluster/work/cotterell/knobelf/data/data_gpt2neo.json",
            load_size=load_size
        )


main()
