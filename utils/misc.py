# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
from collections import OrderedDict
from typing import Tuple
import collections.abc

import torch
import torch.distributed as dist
from datasets import template_meta

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def parse_losses(losses) -> Tuple[torch.Tensor, OrderedDict]:
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f"{loss_name} is not a tensor or list of tensors")

    loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

    return loss, log_vars

def data2cuda(data):

    if isinstance(data, torch.Tensor):
        batch = data.cuda(non_blocking=True)
        return batch
    elif isinstance(data, collections.abc.Mapping):
        return {key: data2cuda(data[key]) for key in data}
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
        return [data2cuda(d) for d in data]
    else:
        raise TypeError

def build_dataset_class_tokens(text_transform, template_set, classnames):

    tokens = []
    templates = template_meta[template_set]
    for classname in classnames:
        # format with class
        tokens.append(torch.stack([text_transform(template.format(classname)) for template in templates]))
    # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
    tokens = torch.stack(tokens)

    return tokens