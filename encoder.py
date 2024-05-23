import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import AddAndNorm, FeedForward
from .attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        attention_hidden_dim: int,  # This is the embedding dimensions
        ffn_hidden_dim: int,
        nheads: int,
    ):
        super().__init__()
        self.attention_hidden_dim = attention_hidden_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.nheads = nheads

        self.attention_layer = MultiHeadAttention(
            hidden_dim=attention_hidden_dim, nheads=nheads
        )
        self.add_and_norm = AddAndNorm()
        self.ffn = FeedForward(attention_hidden_dim, ffn_hidden_dim)

    def forward(self, x: torch.Tensor):
        attention_result = self.attention_layer(encoder_outputs=x)
        attention_result = self.add_and_norm(x, attention_result)
        ffn_result = self.ffn(attention_result)
        ffn_result = self.add_and_norm(attention_result, ffn_result)
        return ffn_result


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        attention_hidden_dim: int,
        ffn_hidden_dim: int,
        nheads: int,
        n_encoder_layers: int = 6,
    ):
        super().__init__()
        self.encoder_layer_sequence = [
            TransformerEncoderLayer(
                attention_hidden_dim=attention_hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                nheads=nheads,
            )
            for _ in range(n_encoder_layers)
        ]

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor):
        return self.encoder_layer_sequence.forward(x)
