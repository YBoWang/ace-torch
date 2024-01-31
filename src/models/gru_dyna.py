import torch
import torch.nn as nn
import src.algorithm.helper as h
from torch import jit
from src.models.rnns import NormGRUCell
import torch.nn.functional as F
from copy import deepcopy
from torch.distributions.normal import Normal


class DGruDyna(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = deepcopy(cfg)
        self.prior_mlp = h.mlp_norm(cfg.hidden_dim, cfg.mlp_dim, cfg.latent_dim, cfg)
        if cfg.norm_cell:
            self.gru_cell = NormGRUCell(cfg.latent_dim+cfg.action_dim, cfg.hidden_dim)
        else:
            self.gru_cell = nn.GRUCell(cfg.latent_dim+cfg.action_dim, cfg.hidden_dim)

    # prepare the init hidden state for gru cell
    def init_hidden_state(self, batch_size, device):
        return torch.zeros((batch_size, self.cfg.hidden_dim), device=device)

    def forward(self, obs_embed, action, h_prev):
        x = torch.cat([obs_embed, action], dim=-1)
        h = self.gru_cell(x, h_prev)
        z_pred = self.prior_mlp(h)
        return z_pred, h


class OneStepDyna(jit.ScriptModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = deepcopy(cfg)
        self.fc1 = nn.Linear(cfg.hidden_dim+cfg.action_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim+cfg.action_dim, cfg.latent_dim)
        self.fc3 = nn.Linear(cfg.latent_dim+cfg.action_dim, cfg.latent_dim)
        self.apply(h.orthogonal_init)

    # one_step prediction model takes hidden state and action to predict next latent state
    def forward(self, hidden, a):
        a = a.detach()
        x = torch.cat([hidden.detach(), a], dim=-1)
        h1 = F.elu(self.fc1(x))
        h1 = torch.cat([h1, a], dim=-1)
        h2 = F.elu(self.fc2(h1))
        h2 = torch.cat([h2, a], dim=-1)
        z_mean = self.fc3(h2)
        dist = Normal(z_mean, torch.ones_like(z_mean).cuda())
        dist = torch.distributions.Independent(dist, 1)
        return dist
