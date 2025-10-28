import dataclasses
import torch
from torch import nn as nn
from torch.nn import functional as F
import os

@dataclasses.dataclass
class ReactEmbedConfig:
    p_dim: int
    m_dim: int
    shared_dim: int
    n_layers: int
    hidden_dim: int
    dropout: float
    normalize_last: int = 1

    def save_to_file(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            for field in dataclasses.fields(self):
                f.write(f"{field.name}={getattr(self, field.name)}\n")

    @staticmethod
    def load_from_file(file_name):
        d = {}
        with open(file_name) as f:
            for line in f.readlines():
                k, v = line.strip().split("=")
                try:
                    d[k] = int(v)
                except ValueError:
                    d[k] = float(v)
        return ReactEmbedConfig(**d)


def get_mlp_layers(dims, dropout=0.0):
    """Creates a simple MLP with ReLU activations"""
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 2):
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        layers.add_module(f"relu_{i}", torch.nn.ReLU())
        if dropout > 0:
            layers.add_module(f"dropout_{i}", torch.nn.Dropout(dropout))
    # Add final layer without activation
    layers.add_module(f"linear_final", torch.nn.Linear(dims[-2], dims[-1]))
    return layers


class EnhancementModule(nn.Module):
    """
    This is the ReactEmbed module, consisting of P2U and M2U MLPs
    as described in the paper (Section 3.1.2).
    """
    def __init__(self, config: ReactEmbedConfig):
        super(EnhancementModule, self).__init__()
        self.config = config
        if config.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        
        # P2U: Protein to Unified
        p_dims = [config.p_dim] + [config.hidden_dim] * (config.n_layers - 1) + [config.shared_dim]
        self.p_to_shared = get_mlp_layers(p_dims, config.dropout)
        
        # M2U: Molecule to Unified
        m_dims = [config.m_dim] + [config.hidden_dim] * (config.n_layers - 1) + [config.shared_dim]
        self.m_to_shared = get_mlp_layers(m_dims, config.dropout)

    def forward(self, x, entity_type):
        """
        Projects a batch of frozen embeddings (x) into the shared space
        based on their type ('P' or 'M').
        """
        if entity_type == 0 or entity_type=="P":
            x = self.p_to_shared(x)
        elif entity_type == 1 or entity_type=="M":
            x = self.m_to_shared(x)
        else:
            raise ValueError(f"Invalid entity_type: {entity_type}. Must be 'P' or 'M'.")

        if self.config.normalize_last:
            return F.normalize(x, dim=-1)
        return x