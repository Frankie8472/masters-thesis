import gc
import json
import os
import sys

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, \
    set_seed


def main():
    topics = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
    models = [1, 2, 3, 4, 7, 8, 9, 10]#[5,6]#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for model in models:
        for topic in topics:
            for union in [0, 1]:
                combi = "un" if union else "is"
                if model == 1:
                    name = f"wiki_nt-gpt2_nt-{combi}-{topic}.txt"
                elif model == 2:
                    name = f"gpt2_nt-wiki_nt-{combi}-{topic}.txt"
                elif model == 3:
                    name = f"gpt2_nt_1-gpt2_nt_2-{combi}-{topic}.txt"
                elif model == 4:
                    name = f"gpt2_nt_2-gpt2_nt_1-{combi}-{topic}.txt"
                elif model == 5:
                    name = f"gpt2_1-gpt2_2-{combi}-{topic}.txt"
                elif model == 6:
                    name = f"gpt2_2-gpt2_1-{combi}-{topic}.txt"
                elif model == 7:
                    name = f"wiki_nt-arxiv-{combi}-{topic}.txt"
                elif model == 8:
                    name = f"arxiv-wiki_nt-{combi}-{topic}.txt"
                elif model == 9:
                    name = f"gpt2_nt-arxiv-{combi}-{topic}.txt"
                else:
                    name = f"arxiv-gpt2_nt-{combi}-{topic}.txt"
                print(
                    f"bsub -N -W 24:00 -n 48 -R \"rusage[mem=2666]\" -o ma/log-{name} \"python ma/train_lda.py {union} {model} {topic}\"")


main()
