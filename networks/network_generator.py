import numpy as np
import torch
import torch.nn as nn
import re


def create_seq(modules: list):
    prev_output = None
    def match_str_to_module(module):
        global prev_output
        if isinstance(module, tuple):
            inp, out = module[0], module[1]
            prev_output = out
            return nn.Linear(inp, out)
        
        if module == 'relu':
            return nn.ReLU()
        
        if isinstance(module, nn.Module):
            return module
        
        if module == 'batch_norm_1d':
            return nn.BatchNorm1d(prev_output)
    
    seq_list = []
    for module in modules:
        module = match_str_to_module(module)
        seq_list.append(module)
    
    return nn.Sequential(*seq_list)
