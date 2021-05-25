from functools import reduce

import torch
from torch import nn
from torch.nn import functional as F

from .circuits import batch_kronecker_complex, ParallelRYComplex, EntangleComplex


class VariationalQuantumCircuitModel(nn.Module):

    def __init__(self, n_qubit: int, n_ancilla: int = 1, k: int = 2):
        super().__init__()
        self.n_ancilla = n_ancilla
        self.layers = nn.ModuleList()
        for _ in range(k):
            self.layers.append(ParallelRYComplex(2 * n_qubit + n_ancilla))
            self.layers.append(EntangleComplex(2 * n_qubit + n_ancilla))
        
        self.out = nn.Linear(1, 1)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor):
        ancilla_qubits = []
        for _ in range(self.n_ancilla):
            ancilla = torch.zeros(len(s1), 2).to(s1.device)
            ancilla[:, 0] = 1
            ancilla_qubits.append((ancilla, torch.zeros_like(ancilla)))

        s = reduce(
            lambda x, y: batch_kronecker_complex(x, y), 
            [
                *ancilla_qubits,
                (s1.real, s1.imag), 
                (s2.real, s2.imag),
            ],
        )
        for layer in self.layers:
            s = layer(s)

        s_real, s_imag = s
        d = torch.sum(s_real[:, ::2] ** 2 + s_imag[:, ::2] ** 2, dim=-1, keepdim=True)
        d = torch.sigmoid(self.out(d))
        return d
