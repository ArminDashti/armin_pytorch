import numpy as np
import torch
import torch.nn as nn
import re

def detect_device():
     return "cuda" if torch.cuda.is_available() else "cpu"


def to_numpy(tensor):
    return tensor.cpu().numpy()


def numpy_to_tensor(tensor, device=None):
    pass


def zeros(shape, grad=False, device=None):
    if type(shape) is int:
        return torch.zeros(shape, requires_grad=grad)
    zero_tensor = torch.zeros(*shape, requires_grad=grad)
    return zero_tensor


def ones(shape, grad=False, device=None):
    zero_tensor = torch.ones(*shape, requires_grad=grad)
    return zero_tensor


def rand(shape, dtype='float', grad=False, device=None):
    if dtype == 'int':
        rand_int = torch.randint(0, 10, shape, requires_grad=grad)
        return rand_int
    else:
        rand_float = torch.rand(*shape, requires_grad=grad)
        return rand_float


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)



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


#%%
net = [(50,50), 'relu', 'batch_norm_1d']
create_seq(net)
