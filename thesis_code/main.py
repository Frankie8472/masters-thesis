# Library imports
import numpy as np
import torch
from transformers import pipeline, set_seed
from transformers import AutoModelForCausalLM, GPT2LMHeadModel
from transformers import AutoTokenizer, GPT2Tokenizer

# Seed for reproducability
set_seed(42)

# Tensorflow or Pytorch
platform = "pt"     # "tf" but not configured for that

# Use GPU or CPU
use_gpu = True
device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

# GPT2
model = "gpt2"
document_length = 1000
tokenizer = GPT2Tokenizer.from_pretrained(model)
model = GPT2LMHeadModel.from_pretrained(model, pad_token_id=tokenizer.eos_token_id)
init_text = tokenizer.bos_token

encoded_input = tokenizer.encode(init_text, return_tensors=platform).to(device)
model = model.to(device)

encoded_output = model.generate(encoded_input, max_length=document_length, do_sample=True, top_k=0)
#if len(encoded_output) == document_length:
#    print(f"ERROR: document length reached, which should not be!")
decoded_output = tokenizer.decode(encoded_output[0], skip_special_tokens=True)

print(decoded_output)
