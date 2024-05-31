import torch
import torch.nn as nn


def num_parameters(module):
    num_parameters = 0
    for p in module.parameters():
        num_parameters += p.nelement()
    return num_parameters