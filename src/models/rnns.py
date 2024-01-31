import torch
import torch.jit as jit
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter


class NormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.ln_reset = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_update = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_newval = nn.LayerNorm(hidden_size, eps=1e-3)

    def forward(self, input, state):
        gates_i = self.weight_ih(input)
        gates_h = self.weight_hh(state)
        reset_i, update_i, newval_i = gates_i.chunk(3, 1)
        reset_h, update_h, newval_h = gates_h.chunk(3, 1)

        reset = torch.sigmoid(self.ln_reset(reset_i + reset_h))
        update = torch.sigmoid(self.ln_update(update_i + update_h))
        newval = torch.tanh(self.ln_newval(newval_i + reset * newval_h))
        h = update * newval + (1 - update) * state
        return h


class GRUCell(jit.ScriptModule):
    """Reproduced regular nn.GRUCell, for reference"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(input_size, 3 * hidden_size))
        self.weight_hh = Parameter(torch.randn(hidden_size, 3 * hidden_size))
        self.bias_ih = Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = Parameter(torch.randn(3 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        gates_i = torch.mm(input, self.weight_ih) + self.bias_ih
        gates_h = torch.mm(state, self.weight_hh) + self.bias_hh
        reset_i, update_i, newval_i = gates_i.chunk(3, 1)
        reset_h, update_h, newval_h = gates_h.chunk(3, 1)
        reset = torch.sigmoid(reset_i + reset_h)
        update = torch.sigmoid(update_i + update_h)
        newval = torch.tanh(newval_i + reset * newval_h)
        h = update * newval + (1 - update) * state
        return h

