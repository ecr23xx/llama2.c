import os
import torch
import argparse
import numpy as np

import torch.distributed as dist

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"

def pretokenize(vocab_source, vocab_size):
    assert vocab_source == "custom", "do not support llama2"
    # read text file and pretokenize it
    data_dir = os.path.join(DATA_CACHE_DIR, "tinyshakespeare")
    with open(os.path.join(data_dir, "input.txt")) as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    enc = Tokenizer(os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model"))
    train_ids = enc.encode(train_data, bos=True, eos=False)
    val_ids = enc.encode(val_data, bos=True, eos=False)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    tokdir = os.path.join(DATA_CACHE_DIR, "tinyshakespeare", f"{vocab_source}_tok{vocab_size}")
    os.makedirs(tokdir, exist_ok=True)
    train_ids.tofile(os.path.join(tokdir, 'train.bin'))
    val_ids.tofile(os.path.join(tokdir, 'val.bin'))


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        bin_dir = os.path.join(DATA_CACHE_DIR, "tinyshakespeare", f"{self.vocab_source}_tok{self.vocab_size}")
        bin_name = os.path.join(bin_dir, f"{self.split}.bin")
        assert os.path.exists(bin_name)
        while True:
            # open the dataset for reading but keep it on disk with memmap
            m = np.memmap(bin_name, dtype=np.uint16, mode="r")
            num_batches = len(m) // self.max_seq_len
            num_batches -= 1  # drop the last partial batch
            assert num_batches > 0, "this shard is way too small? investigate."
            ixs = list(range(num_batches))
            for ix in ixs:
                start = ix * self.max_seq_len
                end = start + self.max_seq_len + 1
                # calling .astype will copy the data into a new numpy array, now in RAM
                chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                x = chunk[:-1]
                y = chunk[1:]
                yield x, y


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
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
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    # if args.stage == "download":
    #     download()
    # elif args.stage == "train_vocab":
    #     train_vocab(vocab_size=args.vocab_size)
    if args.stage == "pretokenize":
        pretokenize("custom", vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
