from transformer_code.mha import MultiHeadAttention, PointWiseFeedForwardNetwork
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads).to("cuda")
        self.mha2 = MultiHeadAttention(d_model, num_heads).to("cuda")

        self.ffn = PointWiseFeedForwardNetwork(d_model, dff).to("cuda")

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6).to("cuda")
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6).to("cuda")
        self.layernorm3 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6).to("cuda")

        self.dropout1 = nn.Dropout(p = rate).to("cuda")
        self.dropout2 = nn.Dropout(p = rate).to("cuda")
        self.dropout3 = nn.Dropout(p = rate).to("cuda")

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
            
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask) 
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2