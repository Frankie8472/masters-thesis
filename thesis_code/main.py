# Library imports
import numpy as np
import torch
from transformers import pipeline, set_seed
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

# Seed for reproducability
t = GPT2Tokenizer.from_pretrained("gpt2")
print(t.model_input_names)

