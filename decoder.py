import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import AddAndNorm, FeedForward
from .attention import MultiHeadAttention


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        attention_hidden_dim: int,  # This is the embedding dimensions
        ffn_hidden_dim: int,
        nheads: int,
        mask: torch.Tensor = None,
    ):
        super().__init__()
        self.attention_hidden_dim = attention_hidden_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.nheads = nheads

        self.self_attention_layer = MultiHeadAttention(
            hidden_dim=attention_hidden_dim, nheads=nheads, mask=mask
        )
        self.cross_attention_layer = MultiHeadAttention(
            hidden_dim=attention_hidden_dim, nheads=nheads
        )
        self.add_and_norm = AddAndNorm()
        self.ffn = FeedForward(attention_hidden_dim, ffn_hidden_dim)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor):
        self_attention_result = self.self_attention_layer(decoder_outputs=x)
        self_attention_result = self.add_and_norm(x, self_attention_result)
        cross_attention_result = self.cross_attention_layer(
            encoder_outputs=encoder_output, decoder_outputs=self_attention_result
        )
        cross_attention_result = self.add_and_norm(
            self_attention_result, cross_attention_result
        )
        feedforward_result = self.ffn(cross_attention_result)
        feedforward_result = self.add_and_norm(
            cross_attention_result, feedforward_result
        )
        return feedforward_result


class TrasnformerDecoder(nn.Module):
    def __init__(
        self,
        attention_hidden_dim: int,  # This is the embedding dimensions
        ffn_hidden_dim: int,
        nheads: int,
        n_encoder_layers: int = 6,
        mask: torch.Tensor = None,
    ):
        self.decoder_layers = [
            TransformerDecoderLayer(
                attention_hidden_dim=attention_hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                nheads=nheads,
                mask=mask,
            )
            for _ in range(n_encoder_layers)
        ]

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor):
        decoder_output = x
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output)
        return decoder_output

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)
