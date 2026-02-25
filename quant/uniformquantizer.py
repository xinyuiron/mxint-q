import torch
import torch.nn as nn

class WeightQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 4,
        group_size: int = 128,
        dimension: str = "per_channel",
        symmetric: bool = False,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.dimension = dimension
        self.symmetric = symmetric

        if self.symmetric:
            self.qmin = -(2 ** (n_bits - 1) - 1)
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** n_bits - 1
        
    def forward(self, x):
        shape = x.shape
        out_features, in_features = shape
        assert self.dimension == "per_channel", "Only per-channel quantization is supported for weights."

        x = x.view(out_features, -1, self.group_size)

        xmax = x.amax(dim = -1, keepdim = True)
        xmin = x.amin(dim = -1, keepdim = True)
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / self.qmax
            scale = scale.clamp(min = 1e-5, max = 1e5)
            zero_point = None
        else:
            scale = (xmax - xmin) / (self.qmax - self.qmin)
            scale = scale.clamp(min = 1e-5, max = 1e5)
            # zero_point = torch.round(-(xmin) / scale).clamp(self.qmin, self.qmax)
            zero_point = torch.round(-(xmin) / scale)
        
        x_int = torch.round(x / scale)
        if zero_point is not None:
            x_int = x_int + zero_point
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if zero_point is not None:
            x_dequant = x_dequant.sub(zero_point)
        x_dequant = x_dequant.mul(scale).view(shape)
        return x_dequant

class ActQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 9,
        group_size: int = 32,
        dimension: str = "per_channel",
    ):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.dimension = dimension

    def forward(self, x, n_bits = None, group_size = None):
        n_bits = n_bits if n_bits is not None else self.n_bits
        group_size = group_size if group_size is not None else self.group_size

        qmin = -(2 ** (n_bits - 1) - 1)
        qmax = 2 ** (n_bits - 1) - 1

        shape = x.shape
        if x.dim() == 3:
            bsz, seq_len, hidden_dim = x.shape
            if self.dimension == "per_token":
                num_groups = hidden_dim // group_size
                assert num_groups * group_size == hidden_dim, "hidden_dim must be divisible by group_size."
                x = x.view(bsz, seq_len, -1, group_size)
                dim = -1

        elif x.dim() == 4:
            bsz, num_heads, seq_len, head_dim = x.shape
            if self.dimension == "per_token":
                num_groups = head_dim // group_size
                assert num_groups * group_size == head_dim, "head_dim must be divisible by group_size."
                x = x.view(bsz, num_heads, seq_len, -1, group_size)
                dim = -1
            elif self.dimension == "per_channel":
                num_groups = seq_len // group_size
                assert num_groups * group_size == seq_len, "seq_len must be divisible by group_size."
                x = x.view(bsz, num_heads, -1, group_size, head_dim)
                dim = -2
        
        else:
            raise NotImplementedError(f"Unsupported input shape {x.shape} for activation quantization.")

        xmax = x.amax(dim = dim, keepdim = True)
        xmin = x.amin(dim = dim, keepdim = True)

        abs_max = torch.max(xmax.abs(), xmin.abs())
        scale = torch.pow(2.0, torch.round(torch.log2(abs_max / qmax)))
        scale = scale.clamp(min = 1e-5, max = 1e5)
        x_int = torch.round(x / scale).clamp(qmin, qmax)
        x_dequant = x_int.mul(scale).view(shape)

        return x_dequant