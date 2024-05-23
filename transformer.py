import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TransformerEncoder
from .decoder import TrasnformerDecoder
from .utils import SinusoidalPositionalEncoding


class VanillaTransformer(nn.Module):
    def __init__(
        self,
        attention_hidden_dim: int,
        ffn_hidden_dim: int,
        nheads: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
    ):
        super().__init__()
        self._reset_parameters()
        self.encoder = TransformerEncoder(
            attention_hidden_dim=attention_hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            nheads=nheads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
        )
        self.decoder = TrasnformerDecoder(
            attention_hidden_dim=attention_hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            nheads=nheads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
        )

    def forward(self, x: torch.Tensor):
        encoder_output = self.encoder(x)
        return encoder_output

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)
