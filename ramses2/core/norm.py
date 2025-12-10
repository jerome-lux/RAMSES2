from functools import partial

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    # normalized_shape MUST be defined when intanciating the Layer
    def forward(self, input: Tensor) -> Tensor:
        input = input.permute(0, 2, 3, 1)
        input = F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        input = input.permute(0, 3, 1, 2)
        return input


class RMSNorm2d(nn.RMSNorm):
    # normalized_shape MUST be defined when intanciating the Layer
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
