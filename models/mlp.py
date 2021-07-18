import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self,inp_dim,output_dim):
    super().__init__()

    self.mlp = nn.Sequential(
        nn.Linear(inp_dim,32),
        nn.ReLU(),
        nn.Linear(32,16),
        nn.ReLU(),
        nn.Linear(16,8),
        nn.ReLU(),
        nn.Linear(8,output_dim)
    )

  def forward(self,x):
    b = x.size(0)
    x = x.view(b,-1)
    return self.mlp(x)
