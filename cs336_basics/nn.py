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
