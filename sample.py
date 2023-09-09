"""
Sample from the trained model with PyTorch
"""
import os
import pickle
import json
from contextlib import nullcontext
import torch
from model import ModelArgs, Transformer
from tokenizer import Tokenizer

from tinystories import get_tokenizer_model_path
from lora import LoraArgs, add_lora, remove_lora
import torch.autograd.profiler as profiler
import torch.nn.utils.parametrize as P

# -----------------------------------------------------------------------------
checkpoint = 'out/ckpt.pt'
start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
# start = "the president of Mexico in 2019"
num_samples = 1 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 0.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = "" # override the tokenizer model path
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = "bfloat16-true"  # float32|bfloat16-mixed|float16-true|bfloat16-true
compile = False # use PyTorch 2.0 to compile the model to be faster
vocab_size = 32000
max_seq_len = 2048
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {"float32": torch.float32, "bfloat16-mixed": torch.bfloat16, "bfloat16-true": torch.bfloat16, "float16-true": torch.float16}[dtype]
ctx = (
    torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    if device_type == "cuda" and "mixed" in dtype
    else nullcontext()
)

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location="cpu")
model_args = dict()
if "model_args" in checkpoint_dict:
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
        model_args[k] = checkpoint_dict["model_args"][k]
    state_dict = checkpoint_dict["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
else:
    params_path = os.path.join(os.path.dirname(checkpoint), "params.json")
    with open(params_path) as f:
        checkpoint_model_args = json.load(f)
    # skip vocab_size and max_seq_len
    for k in ["dim", "n_layers", "n_heads", "multiple_of"]:
        model_args[k] = checkpoint_model_args[k]
    model_args["n_kv_heads"] = model_args["n_heads"]
    model_args["vocab_size"] = vocab_size
    model_args["max_seq_len"] = max_seq_len
    state_dict = checkpoint_dict
    vocab_source = "llama2"
gptconf = ModelArgs(**model_args)
print(gptconf)
model = Transformer(gptconf)

if "lora_args" in checkpoint_dict:
    lora_args = checkpoint_dict["lora_args"]
    add_lora(model, LoraArgs(**lora_args))

if dtype == "bfloat16-true":
    print("conver model to bfloat16 precision")
    model.bfloat16()
elif dtype == "float16-true":
    print("conver model to float16 precision")
    model.half()
else:
    print(f"use {dtype} precision")

missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
if len(missing_keys) > 0 or len(unexpected_keys) > 0:
    print(missing_keys, unexpected_keys)

# print(model.output.weight)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load the tokenizer
vocab_size = gptconf.vocab_size
if tokenizer:
    # a specific tokenizer is provided, use it
    tokenizer_model = tokenizer
else:
    # let's try to find the tokenizer model automatically. bit gross here...
    query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
    tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
# print(tokenizer_model)
enc = Tokenizer(tokenizer_model=tokenizer_model)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = enc.encode(start, bos=True, eos=False)
# print(start_ids)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad(), P.cached():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(enc.decode(y[0].tolist()))
            print('---------------')
