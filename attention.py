import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Add dimensional annotation to all files
class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        hidden_dim: int,  # d_{model} in original paper
        nheads: int,  # h in original paper
        mask: torch.Tensor = None,
    ):
        super().__init__()
        assert hidden_dim % nheads == 0, "hidden_dim should be divisible by nheads"
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.mask = mask

        self.is_masked = not (self.mask is None)
        self.d_k = hidden_dim / nheads
        self.d_v = hidden_dim / nheads

        self.W_qkv = nn.Linear(hidden_dim, self.d_k * 2 + self.d_v, bias=False)
        """
        The formulation for W_qkv is esentially a concatenation of:
        - W^Q, W^K, W^V vertically
        - W_1^Q, W_2^Q, \dots, W_h^Q horizontally
        """  # TODO: Figure out
        self.W_o = nn.Linear(nheads * self.d_v, hidden_dim, bias=False)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_qkv)
        nn.init.xavier_uniform_(self.W_o)

    def _get_self_attention_qkv(self, x: torch.Tensor):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.W_qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, self.nheads, self.d_v + 2 * self.d_k)
        return qkv.chunk(3, dim=-1)

    def _get_cross_attention_qkv(
        self, encoder_output: torch.Tensor, decoder_output: torch.Tensor
    ):
        assert (
            encoder_output.shape[0] == decoder_output.shape[0]
        ), ""  # TODO: write assert message here
        assert (
            encoder_output.shape[2] == decoder_output.shape[2]
        ), ""  # TODO: write assert message here

        batch_size, encoder_seq_len, embed_dim = encoder_output.shape
        decoder_seq_len = decoder_output.shape

        weights_q, weights_kv = self.W_qkv.weight.split(
            [self.hidden_dim, self.hidden_dim * 2]
        )
        q = F.linear(input=decoder_output, weight=weights_q).reshape(
            batch_size, decoder_seq_len, self.nheads, self.d_k
        )
        k, v = (
            F.linear(input=encoder_output, weight=weights_kv)
            .reshape(batch_size, encoder_seq_len, self.nheads, self.d_k + self.d_v)
            .chunk(2, dim=-1)
        )

        return q, k, v

    def scaled_dot_product(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_masked: bool = False,
    ):
        return  # TODO: Implement is_masked as well, and partitions

    def scaled_dot_product_sparse(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_masked: bool = True,  # Toggles the sparsifications
    ):
        pass  # TODO: Implement

    def forward(
        self,
        encoder_outputs: torch.Tensor = None,
        decoder_outputs: torch.Tensor = None,
    ):
        assert not (
            encoder_outputs is None and decoder_outputs is None
        ), "Encoder output or decoder output needs to not be None"

        if not (encoder_outputs is None or decoder_outputs is None):
            q, k, v = self._get_cross_attention_qkv(
                encoder_output=encoder_outputs, decoder_output=decoder_outputs
            )
            attention_outcomes = self.scaled_dot_product(
                q, k, v, is_masked=self.is_masked
            )
            batch_size, seq_len, embed_size = decoder_outputs.shape
        else:
            self_attention_input = encoder_outputs
            if encoder_outputs is None:
                self_attention_input = decoder_outputs
            q, k, v = self._get_self_attention_qkv(x=self_attention_input)
            attention_outcomes = self.scaled_dot_product(
                q, k, v, is_masked=self.is_masked
            )
            batch_size, seq_len, embed_size = self_attention_input.shape

        concat_attention = attention_outcomes.reshape(batch_size, seq_len, embed_size)
        # Shape of cat above follows whoever is using this attention.
        # In cross case, it is decoder; else, it is self_attention_input.
        return self.W_o(concat_attention)
