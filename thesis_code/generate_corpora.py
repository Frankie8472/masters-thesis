import gc
import json
import os
import sys
import random
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, set_seed, TransfoXLTokenizer, TransfoXLLMHeadModel
import pickle


def get_state(cuda_device):
    """
    Helper function for reproducible behavior to set the state in `random`, `numpy`, `torch`.
    Args:
        cuda_device: The cuda device.
    Returns:
        random_state: The current state.
        numpy_state: The current state.
        torch_state: The current state.
        cuda_state: The current state.
    """
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(cuda_device)
    return [random_state, numpy_state, torch_state, cuda_state]


def set_state(state, cuda_device):
    """
    Helper function for reproducible behavior to set the state in `random`, `numpy`, `torch`.

    Args:
        state: Array with the following entries
            0: random_state: The state to set.
            1: numpy_state: The state to set.
            2: torch_state: The state to set.
            3: cuda_state: The state to set.
        cuda_device: The cuda device to set.
    """
    random_state, numpy_state, torch_state, cuda_state = state[0], state[1], state[2], state[3]
    random.setstate(random_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    torch.cuda.set_rng_state(cuda_state, cuda_device)
    return


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
    try:
        # prepare filepaths
        split = save_path.split('/')
        rng_path = '/'.join(split[:-1])+f"/rng/{split[-1][:-5]}_rng.pickle"

        # create folder if neccessary
        os.makedirs(os.path.dirname(rng_path), exist_ok=True)

        # load random state
        if os.path.isfile(save_path):
            with open(save_path, 'r') as file:
                decoded_output = json.load(file)
                if len(decoded_output) >= corpus_size:
                    raise AssertionError(">> ERROR: File has already reached desired size")

            if os.path.isfile(rng_path):
                with open(rng_path, "rb") as file:
                    state = pickle.load(file)
                    set_state(state, device)
            else:
                raise FileNotFoundError(">> ERROR: File exists but has no rng state file, therefore cannot be enhanced. Delete file for overwriting")

            print(f">> Enhancing corpus from {len(decoded_output)} to {corpus_size} documents")
        else:
            decoded_output = []

        # get tokenizer
        tokenizer = tokenizer_model.from_pretrained(tokenizer_name)

        # get model
        model = lm_model.from_pretrained(model_name)

        # set parameters for document generation
        max_document_length = max_document_length if max_document_length is not None else tokenizer.model_max_length
        if pad_token_id is not None:
            if pad_token_id == 'eos_token_id':
                pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError(">> ERROR: Undefinded/unimplemented pad_token_id")

        if bos_token_id is not None:
            if bos_token_id == 'eos_token_id':
                bos_token_id = tokenizer.eos_token_id
            else:
                raise ValueError(">> ERROR: Undefinded/unimplemented bod_token_id")

        if eos_token_id is not None:
            if eos_token_id == 'eos_token_id':
                eos_token_id = tokenizer.eos_token_id
            else:
                raise ValueError(">> ERROR: Undefinded/unimplemented bod_token_id")

        # print(f">> EOS: {tokenizer.eos_token} | BOS: {tokenizer.bos_token} | UNK: {tokenizer.unk_token}")

        # load the model to the device (neccessary when using the gpu)
        model = model.to(device)

        # initialize progress bar
        with tqdm(total=corpus_size - len(decoded_output)) as pbar:

            # repeat doc generation more times than neccessary with break condition because there can be empty docs
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

                # save the intermediate resulting documents to the return list
                for j in range(load_size):
                    out_tmp = tokenizer.decode(encoded_output[j], skip_special_tokens=True)
                    if out_tmp != "" and len(decoded_output) != corpus_size:
                        decoded_output.append(out_tmp)
                        pbar.update(1)
                    if len(decoded_output) == corpus_size:
                        break

                # check break condition
                if len(decoded_output) == corpus_size:
                    print(f">> Expected Size reached after {i}*{load_size} iterations")
                    break

        assert len(decoded_output) == corpus_size, ">> ERROR: Expected Size not reached, to many empty strings"

        # save generated corpus
        with open(save_path, 'w') as file:
            json.dump(decoded_output, file, indent=2)

        # save random state for further generation without having to start from scratch
        with open(rng_path, "wb") as file:
            pickle.dump(get_state(device), file)

    # cleanup system
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main():
    """
    Command:
        python generate_corpora.py [data_path] [model] [sampling_method] [index] [corpus_size]

        Models:
            gpt2, gpt2-wiki_nt, gpt2-wiki, gpt2-wiki-integrated, trafo_xl, trafo_xl-wiki, trafo_xl-wiki-integrated, trafo_xl-wiki_nt
        Sampling_method:
            multinomial, top_p, typ_p
        index:
            For different samples (changes the seed, 0 > 42, 1 > 1337)
        corpus_size:
            Number of documents you want in the corpus in total

    Run Example:
        python generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt multinomial 0 50000
    """

    # autodetect best device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # define loadsize for how many documents fit into the gpu memory, has impact on how fast the documents are generated
    # eventually needs to be tweaked for different gpus
    load_size = 50

    assert len(sys.argv) >= 6, ">> ERROR: Wrong input arguments"

    # get input parameters and cast to matching type
    data_folder_path = sys.argv[1]
    model = sys.argv[2]
    sampling = sys.argv[3]
    index = int(sys.argv[4])
    corpus_size = int(sys.argv[5])

    # preprocessing of the datapaths
    if data_folder_path[-1] != "/":
        data_folder_path += "/"

    dataset_path = data_folder_path + "datasets/"
    if index == 0:
        dataset_path += "dataset1-"
        set_seed(42)
    elif index == 1:
        set_seed(1337)
        dataset_path += "dataset2-"
    else:
        raise ValueError(f">> ERROR: This index has not yet been implemented")

    dataset_path += model

    # define parameters for doc generation
    max_document_length = None
    bos_token_id = None
    pad_token_id = None
    eos_token_id = None
    top_p = 1.0
    typ_p = 1.0

    # define sampling identifier for datapath
    if sampling == "multinomial":
        pass
    elif sampling == "typ_p":
        dataset_path += "-" + sampling
        typ_p = 0.2
    elif sampling == "top_p":
        dataset_path += "-" + sampling
        top_p = 0.9
    else:
        raise ValueError(">> ERROR: This sampling method has not yet been implemented")

    # setting model parameters according to model identifier
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
        max_document_length = 1024
        tokenizer_name = "transfo-xl-wt103"
        model_path = "transfo-xl-wt103"
        bos_token_id = 'eos_token_id'
        pad_token_id = 'eos_token_id'
        tokenizer_model = TransfoXLTokenizer
        lm_model = TransfoXLLMHeadModel
    elif model == "trafo_xl-wiki" or model == "trafo_xl-wiki-integrated" or model == "trafo_xl-wiki_nt":
        max_document_length = 1024
        tokenizer_name = "transfo-xl-wt103"
        model_path = f"{data_folder_path}model-{model}"
        bos_token_id = 'eos_token_id'
        pad_token_id = 'eos_token_id'
        tokenizer_model = TransfoXLTokenizer
        lm_model = TransfoXLLMHeadModel
    else:
        raise ValueError(">> ERROR: This model has not yet been implemented")

    # call function for corpus creation
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
