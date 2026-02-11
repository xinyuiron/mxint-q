import torch
import torch.nn as nn
import torch.nn.functional as F
from quant.uniformquantizer import ActQuantizer, WeightQuantizer

class QuantLinear(nn.Module):
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        super().__init__()

        self.fwd_func = F.linear
        self.register_buffer("weight", org_module.weight)

        if org_module.bias is not None:
            self.register_buffer("bias", org_module.bias)
        else:
            self.bias = None
        
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.weight_quantizer = WeightQuantizer(**weight_quant_params)
        self.act_quantizer = ActQuantizer(**act_quant_params)
    
    def forward(self, input: torch.Tensor):
        input = self.act_quantizer(input)
        out = self.fwd_func(input, self.weight, self.bias)
        return out