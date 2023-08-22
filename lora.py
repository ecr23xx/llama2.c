"""
References: https://github.com/cccntu/minLoRA/blob/main/minlora/model.py
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P


@dataclass
class LoraArgs:
    rank: int = 8
    alpha: float = 8
    dropout_p: float = 0.00
    target_modules: list = field(default_factory=list)


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, weight_type, args: LoraArgs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_alpha = args.alpha
        self.lora_rank = args.rank
        self.lora_dropout_p = args.dropout_p
        self.weight_type = weight_type
        assert weight_type in ["emb", "linear"]

        self.lora_A = nn.Parameter(torch.zeros(in_features, self.lora_rank))
        self.lora_B = nn.Parameter(torch.zeros(self.lora_rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = self.lora_alpha / self.lora_rank
        self.dropout = nn.Dropout(self.lora_dropout_p)
        self.register_buffer(
            "lora_dropout_mask", torch.ones(in_features, 1, dtype=self.lora_A.dtype)
        )

    def forward(self, weight):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        dropout_lora_A = self.dropout(self.lora_dropout_mask) * self.lora_A
        delta = dropout_lora_A @ self.lora_B
        if self.weight_type == "linear":
            # linear.weight.shape: [out, in]
            return weight + self.scaling * delta.T
        elif self.weight_type == "emb":
            # emb.weight.shape: [in, out]
            return weight + self.scaling * delta
        else:
            raise NotImplementedError

    def __repr__(self):
        return f"LoRA(in_features={self.in_features}, out_features={self.out_features}, weight_type={self.weight_type}, lora_rank={self.lora_rank}, lora_alpha={self.lora_alpha}, lora_dropout_p={self.lora_dropout_p}"


def add_lora(model, args: LoraArgs):
    for idx, (name, layer) in enumerate(model.named_modules()):
        if name.split(".")[-1] not in args.target_modules:
            continue
        elif isinstance(layer, nn.Linear):
            print("add lora to", name)
            out_features, in_features = layer.weight.shape
            P.register_parametrization(
                layer, "weight",
                LoRA(in_features, out_features, "linear", args),
            )
        elif isinstance(layer, nn.Embedding):
            print("add lora to", name)
            in_features, out_features = layer.weight.shape
            P.register_parametrization(
                layer, "weight",
                LoRA(in_features, out_features, "emb", args),
            )


def merge_lora(model):
    def merge_lora_layer(layer):
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                # set leave_parametrized = True
                P.remove_parametrizations(layer, attr_name, leave_parametrized=True)
    model.apply(merge_lora_layer)


def remove_lora(model):
    def remove_lora_layer(layer):
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                P.remove_parametrizations(layer, attr_name, leave_parametrized=False)
    model.apply(remove_lora_layer)


def name_is_lora(name):
    return (
        len(name.split(".")) >= 4
        and (name.split(".")[-4]) == "parametrizations"
        and name.split(".")[-1] in ["lora_A", "lora_B"]
    )


def get_lora_params(model, print_shapes=False):
    for n, p in model.named_parameters():
        if name_is_lora(n):
            if print_shapes:
                print(n, p.shape)
            yield p


def get_lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if name_is_lora(k)}
