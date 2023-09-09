import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"
IGNORE_INDEX = -1

def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


def prepare_sample(example, enc, mask_inputs=False):
    """Prepare a single sample for training."""
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = enc.encode(full_prompt, bos=True, eos=False)
    encoded_full_prompt_and_response = enc.encode(full_prompt_and_response, bos=True, eos=True)

    # The labels are the full prompt with response. Optionally, the prompt can be masked out.
    # But empirically, we do not mask the prompt https://github.com/Lightning-AI/lit-llama/issues/232
    if mask_inputs:
        labels = encoded_full_prompt_and_response.clone()
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX
        return encoded_full_prompt_and_response, labels

    return encoded_full_prompt_and_response


def pretokenize(vocab_source, vocab_size):
    # read text file and pretokenize it
    data_dir = os.path.join(DATA_CACHE_DIR, "alpaca")
    with open(os.path.join(data_dir, "alpaca_data_cleaned_archive.json")) as f:
        data = json.load(f)
    n = len(data)
    # TODO: does it really random?
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    enc = Tokenizer(os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model"))

    train_ids = [prepare_sample(sample, enc) for sample in tqdm(train_data)]
    val_ids = [prepare_sample(sample, enc) for sample in tqdm(val_data)]
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    tokdir = os.path.join(data_dir, f"{vocab_source}_tok{vocab_size}")
    os.makedirs(tokdir, exist_ok=True)
    torch.save(train_ids, os.path.join(tokdir, "train.pt"))
    torch.save(val_ids, os.path.join(tokdir, "val.pt"))


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        bin_dir = os.path.join(DATA_CACHE_DIR, "alpaca", f"{self.vocab_source}_tok{self.vocab_size}")
        bin_name = os.path.join(bin_dir, f"{self.split}.pt")
        assert os.path.exists(bin_name), f"File {bin_name} does not exist"
        self.m = torch.load(bin_name)

    def __iter__(self):
        while True:
            num_batches = len(self.m)
            ixs = list(range(num_batches))
            for ix in ixs:
                end = self.max_seq_len + 1
                # calling .astype will copy the data into a new numpy array, now in RAM
                chunk = torch.tensor((self.m[ix][:end])).long()
                x = chunk[:-1]
                y = chunk[1:]
                yield x, y


class Task:
    @staticmethod
    def collate_fn(batch):
        # pad sequences
        x, y = zip(*batch)
        x = pad_sequence(x, batch_first=True, padding_value=0)
        y = pad_sequence(y, batch_first=True, padding_value=-1)
        return x, y

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers,
            collate_fn=Task.collate_fn
        )

        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab"])
    parser.add_argument("--vocab_source", type=str, default="llama2", choices=["llama2", "custom"])
    parser.add_argument("--vocab_size", type=int, default=32000, help="pretokenization vocab size. 32000 = use Llama 2 tokenizer.")
    parser.add_argument("--max-length", type=int, default=256, help="pretokenization vocab size. 32000 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    # if args.stage == "download":
    #     download()
    # elif args.stage == "train_vocab":
    #     train_vocab(vocab_size=args.vocab_size)
    if args.stage == "pretokenize":
        pretokenize(vocab_source=args.vocab_source, vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
