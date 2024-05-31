import numpy as np
import torch
import torch.nn as nn
import re

def detect_device():
     return "cuda" if torch.cuda.is_available() else "cpu"