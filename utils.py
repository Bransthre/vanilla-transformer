import torch
import torch.nn as nn
import torch.nn.functional as F

"""
- Positional Encoding
- Other eventually useful things
"""


class AddAndNorm(nn.Module):
    def __init__():
        super().__init__()

    def forward(x, fx):
        return F.normalize(x + fx)


class FeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,  # This is the same as output_dim
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.model_sequence = nn.Sequential(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            ]
        )

    def forward(self, x):
        fx = self.model_sequence.forward(x)
        return AddAndNorm(x, fx)


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(
        self,
        max_seq_len: int,
        hidden_dim: int,
    ):
        self.position_encodings = torch.zeros((max_seq_len, hidden_dim))

        dimensions = torch.arange(hidden_dim)
        # Working with sines
        sine_positions = torch.arange(0, max_seq_len, 2).repeat(hidden_dim, 1)
        sine_dimensions = dimensions.unsqueeze(1).repeat(1, sine_positions.shape[0])
        sine_values = torch.sin(sine_positions / 10000 ^ (sine_dimensions / hidden_dim))
        # Working with cosines
        cosine_positions = torch.arange(1, max_seq_len, 2).repeat(hidden_dim, 1)
        cosine_dimensions = dimensions.unsqueeze(1).repeat(1, cosine_positions.shape[0])
        cosine_values = torch.cos(
            cosine_positions / 10000 ^ (cosine_dimensions / hidden_dim)
        )

        self.position_encodings[:, 0::2] = sine_values
        self.position_encodings[:, 1::2] = cosine_values

    def forward(self, x):
        # Adds sequence to sequence, remember sequences are (max_len, hidden_dim)
        assert (
            x.shape == self.position_encodings.shape
        ), f"Shape of x is {x.shape}, not equal to {self.position_encodings.shape}"
        return x + self.position_encodings
