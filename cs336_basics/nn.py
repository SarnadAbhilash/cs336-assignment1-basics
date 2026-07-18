import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        # allocate empty tensor of shape (d_out, d_in)
        self.weight = nn.Parameter(
            torch.empty(d_out, d_in, device=device, dtype=dtype)
        )
        # 2. compute std per spec : sigma squared = 2 / (d_in + d_out) ** 0.5
        std = (2.0 / (d_in + d_out)) ** 0.5
        # 3. fill it in place truncated at [-3*std, 3*std]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        # x: (..., d_in)  ->  return (..., d_out)
        # one line: matmul x with the transpose of the weight
        # y = x @ W.T
        return x @ self.weight.T

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype = None):
        super().__init__()
        # epsilon which is a small constant to avoid division by zero
        self.eps = eps
        # learnable gain parameter that helps in rescaling each feature
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x):
        # cast to float32 to avoid numerical instability and then back to the original dtype
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # x: (..., , d_model) --> (..., , d_model)
        # rms normalization: divide by the root mean square of the input

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x/rms * self.weight
        return result.to(in_dtype)
