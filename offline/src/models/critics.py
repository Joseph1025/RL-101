"""
Critics for offline RL (adapted from dppo).
"""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """Simple MLP with configurable activation."""
    
    def __init__(
        self,
        dims: List[int],
        activation: str = "Mish",
        out_activation: str = "Identity",
        use_layernorm: bool = False,
    ):
        super().__init__()
        act_cls = getattr(nn, activation) if activation != "Identity" else nn.Identity
        out_act_cls = getattr(nn, out_activation) if out_activation != "Identity" else nn.Identity
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_layernorm and i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
            else:
                layers.append(out_act_cls())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class DoubleCritic(nn.Module):
    """Double Q-network for TD3/SAC/IQL style algorithms."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "Mish",
        use_layernorm: bool = False,
    ):
        super().__init__()
        dims = [state_dim + action_dim] + hidden_dims + [1]
        
        self.Q1 = MLP(dims, activation=activation, use_layernorm=use_layernorm)
        self.Q2 = MLP(dims, activation=activation, use_layernorm=use_layernorm)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        return self.Q1(x).squeeze(-1), self.Q2(x).squeeze(-1)
    
    def q1(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        return self.Q1(x).squeeze(-1)


class ValueNetwork(nn.Module):
    """State-only value network V(s) for IQL."""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "Mish",
        use_layernorm: bool = False,
    ):
        super().__init__()
        dims = [state_dim] + hidden_dims + [1]
        self.V = MLP(dims, activation=activation, use_layernorm=use_layernorm)
    
    def forward(self, state: torch.Tensor):
        return self.V(state).squeeze(-1)
