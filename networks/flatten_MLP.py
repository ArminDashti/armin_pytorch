# From https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py

import torch
import torch.nn as nn
from armin_utils.utils import MLP


class flatten_MLP(MLP.MLP):
    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)