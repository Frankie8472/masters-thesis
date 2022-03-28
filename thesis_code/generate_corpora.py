import gc
import json
import os
import sys
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, set_seed, TransfoXLTokenizer, TransfoXLLMHeadModel


def create_corpus(
        tokenizer_name="gpt2",
        model_name="gpt2",
        max_document_length=None,
        device="cpu",
        corpus_size=1,
        tokenizer_model=AutoTokenizer,
        lm_model=AutoModelForCausalLM,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        save_path="data/test",
        load_size=1,
        top_p=1.0,
        typ_p=1.0
    ):
    r"""
    Generates sequences/documents/a corpus for models with a language modeling head.

    Parameters:
        tokenizer_name (`str`, *optional*, defaults to "gpt2
            The name of the pre-trained tokenizer: openai-gpt, gpt2-small, gpt2, gpt2-large, gpt2-xl, transfo-xl-wt103, EleutherAI/gpt-neo-2.7B, ctrl
        model_name (`str`, *optional*, defaults to "gpt2"):
            The model name of the pre-trained model: openai-gpt, gpt2-small, gpt2, gpt2-large, gpt2-xl, transfo-xl-wt103, EleutherAI/gpt-neo-2.7B, ctrl
        max_document_length (`int`, *optional*, defaults to None):
            The maximum document length, normally set to tokenizer.max_length
        device (`str`, *optional*, defaults to "cpu"):
            The device the computations commence "cpu" or "cuda"
        corpus_size (`int`, *optional*, defaults to 1):
            The corpus size to be generated (number of documents)
        tokenizer_model (`PreTrainedTokenizer`, *optional*, defaults to AutoTokenizer):
            The pre-trained tokenizer class
        lm_model (`PreTrainedModel`, *optional*, defaults to AutoModelForCausalLM):
            The pre-trained model class with language modeling head
        bos_token_id (`str`, *optional*, defaults to None)
            Id of the BOS token. Some models need the pad token set to the eos token (see hugging face documentation).
        pad_token_id (`str`, *optional*, defaults to None)
            Id of the padding token. Some models need the pad token set to the eos token (see hugging face documentation).
        eos_token_id (`str`, *optional*, defaults to None)
            Id of the EOS token. Some models need the pad token set to the eos token (see hugging face documentation).
        save_path ('str', *optional*, defaults to "data/test")
            Save path for the generated corpus
        load_size (`int`, *optional*, defaults to 1)
            Load size of how many documents can be calculated in the same iteration (depends on (GPU) memory and max_document_length)
        top_p (`float`, *optional*, defaults to 1.0)
        typ_p (`float`, *optional*, defaults to 1.0)
            Multinomial sampling: top_p = 1.0, typ_p = 1.0
            Top_p sampling: typ_p = 1.0, top_p = ]0.0, 1.0[
            Typ_p sampling: top_p = 1.0, typ_p = ]0.0, 1.0[
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

    if bos_token_id is not None:
        if bos_token_id == 'eos_token_id':
            bos_token_id = tokenizer.eos_token_id
        else:
            print("ERROR: Undefinded/unimplemented bod_token_id")

    if eos_token_id is not None:
        if eos_token_id == 'eos_token_id':
            eos_token_id = tokenizer.eos_token_id
        else:
            print("ERROR: Undefinded/unimplemented bod_token_id")

    print(f"EOS: {tokenizer.eos_token} | BOS: {tokenizer.bos_token} | UNK: {tokenizer.unk_token}")

    model = model.to(device)

    decoded_output = []

    with tqdm(total=corpus_size) as pbar:
        for i in range(0, 4*corpus_size):
            encoded_output = model.generate(
                # all parameters have to be set as otherwise the config of the pretrained model will be taken
                input_ids=None,
                max_length=max_document_length,
                do_sample=True,                         # False implies Greedy search
                early_stopping=False,
                num_beams=1,                            # 1 deactivates beam_search
                temperature=1.0,                        # 1.0 deactivates temperature
                top_k=0,                                # 0 deactivates top_k sampling
                top_p=top_p,                            # 1.0 deactivates top_p (nucleus) sampling  using 0.9
                typical_p=typ_p,                        # 1.0 deactivates typical_p sampling        using 0.2
                repetition_penalty=1.0,                 # 1.0 deactivates repetition_penalty
                pad_token_id=pad_token_id,              # For open-end generation set to eos_token_id
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                length_penalty=1.0,                     # 1.0 deactivates length_penalty
                no_repeat_ngram_size=0,                 # 0 deactivates no_repeat_ngram_size
                encoder_no_repeat_ngram_size=0,         # 0 deactivates encoder_no_repeat_ngram_size
                num_return_sequences=load_size,         # The number of independently computed returned sequences for each element in the batch. No input means batch size of one.
                num_beam_groups=1,
                output_scores=False,                    # Will be important if you want the prediction scores!
            )

            for j in range(load_size):
                out_tmp = tokenizer.decode(encoded_output[j], skip_special_tokens=True)
                if out_tmp != "" and len(decoded_output) != corpus_size:
                    decoded_output.append(out_tmp)
                    pbar.update(1)
                if len(decoded_output) == corpus_size:
                    break

            if len(decoded_output) == corpus_size:
                print(f">> Expected Size reached after {i}*{load_size} iterations")
                break

    if len(decoded_output) != corpus_size:
        print(">> ERROR: Expected Size not reached, to many empty strings")
        gc.collect()
        torch.cuda.empty_cache()
        return

    with open(save_path, 'w') as file:
        json.dump(decoded_output, file, indent=2)

    gc.collect()
    torch.cuda.empty_cache()


def main():
    """
    Command:
        python generate_corpora.py [data_path] [model] [sampling_method] [index]

        Models:
            gpt2, gpt2-wiki_nt, gpt2-wiki, gpt2-wiki-integrated, trafo_xl, trafo_xl-wiki, trafo_xl-wiki-integrated, trafo_xl-wiki_nt
        Sampling_method:
            multinomial, top_p, typ_p
        index:
            For different samples (changes the seed, 0 > 42, 1 > 1337)

    Run Example:
        python generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt multinomial 0
    """

    corpus_size = 100000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_size = 100

    if len(sys.argv) < 5:
        print("ERROR: Wrong input arguments")
        return

    data_folder_path = sys.argv[1]
    model = sys.argv[2]
    sampling = sys.argv[3]
    index = int(sys.argv[4])

    if data_folder_path[-1] != "/":
        data_folder_path += "/"

    dataset_path = data_folder_path
    if index == 0:
        dataset_path += "dataset1-"
        set_seed(42)
    elif index == 1:
        set_seed(1337)
        dataset_path += "dataset2-"
    else:
        print(f">> ERROR: This index has not yet been implemented")
        return

    dataset_path += model

    max_document_length = None
    bos_token_id = None
    pad_token_id = None
    eos_token_id = None
    top_p = 1.0
    typ_p = 1.0

    if sampling == "multinomial":
        pass
    elif sampling == "typ_p":
        dataset_path += "-" + sampling
        typ_p = 0.2
    elif sampling == "top_p":
        dataset_path += "-" + sampling
        top_p = 0.9
    else:
        print(">> ERROR: This sampling method has not yet been implemented")
        return

    if model == "gpt2":
        tokenizer_name = "gpt2"
        model_path = "gpt2"
        pad_token_id = 'eos_token_id'
        tokenizer_model = GPT2Tokenizer
        lm_model = GPT2LMHeadModel
    elif model == "gpt2-wiki_nt" or model == "gpt2-wiki" or model == "gpt2-wiki-integrated":
        tokenizer_name = "gpt2"
        model_path = f"{data_folder_path}model-{model}"
        pad_token_id = 'eos_token_id'
        tokenizer_model = GPT2Tokenizer
        lm_model = GPT2LMHeadModel
    elif model == "trafo_xl":
        tokenizer_name = "transfo-xl-wt103"
        model_path = "transfo-xl-wt103"
        bos_token_id = 'eos_token_id'
        pad_token_id = 'eos_token_id'
        tokenizer_model = TransfoXLTokenizer
        lm_model = TransfoXLLMHeadModel
    elif model == "trafo_xl-wiki" or model == "trafo_xl-wiki-integrated" or model == "trafo_xl-wiki_nt":
        tokenizer_name = "transfo-xl-wt103"
        model_path = f"{data_folder_path}model-{model}"
        bos_token_id = 'eos_token_id'
        pad_token_id = 'eos_token_id'
        tokenizer_model = TransfoXLTokenizer
        lm_model = TransfoXLLMHeadModel
    else:
        print(">> ERROR: This model has not yet been implemented")
        return

    create_corpus(
        tokenizer_name=tokenizer_name,
        model_name=model_path,
        max_document_length=max_document_length,
        device=device,
        corpus_size=corpus_size,
        tokenizer_model=tokenizer_model,
        lm_model=lm_model,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        save_path=f"{dataset_path}.json",
        load_size=load_size,
        top_p=top_p,
        typ_p=typ_p
    )


if __name__ == "__main__":
    main()
