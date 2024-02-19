from transformer_code.mha import MultiHeadAttention, PointWiseFeedForwardNetwork
import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff).to("cuda")

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6).to("cuda")
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6).to("cuda")

        self.dropout1 = nn.Dropout(p = rate).to("cuda")
        self.dropout2 = nn.Dropout(p = rate).to("cuda")

    def forward(self, x, mask=None):
        x = torch.squeeze(x, 0)
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


