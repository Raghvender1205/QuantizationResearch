import torch
import torch.nn as nn
from torchvision.models import vit_b_16

# Load Model
model = vit_b_16()
# Determine range of weights and activations
min_val, max_val = float('-inf'), float('inf')
for name, param in model.named_parameters():
    if param.requires_grad:
        min_val = min(min_val, param.min().item())
        max_val = max(max_val, param.max().item())

# Quantize weights and activations
n_bits = 8
max_range = 2 ** n_bits - 1 # Max value that can be represented
scale = (max_val - min_val) / max_range
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = (param.data - min_val) / scale
        param.data = param.data.round()
        param.data = param.data * scale + min_val

print(model)