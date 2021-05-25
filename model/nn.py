import torch
from torch import nn


class NeuralNetworkModel(nn.Module):
    
    def __init__(self, n_qubit: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.layers = nn.Sequential(
            nn.Linear(2 ** n_qubit * 2, 2 ** n_qubit * 2).to(torch.cfloat),
            # nn.ReLU(),
            # nn.Linear(2 ** n_qubit * 4, 2 ** n_qubit * 4),
            # nn.ReLU(),
            # nn.Linear(2 ** n_qubit * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, s1: torch.Tensor, s2: torch.Tensor):
        s = torch.cat([s1, s2], dim=-1)
        # s_r = torch.cat([s.real, s.imag], dim=-1)
        s_r = s
        d = self.layers(s_r)
        return d
