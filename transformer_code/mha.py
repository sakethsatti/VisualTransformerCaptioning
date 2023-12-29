import torch
import torch.nn as nn

def create_padding_mask(seq):
        seq = torch.eq(seq, 0).float()
        return seq.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    
def create_look_ahead_mask(size):
    mask = 1 - torch.tril(torch.ones(size, size))
    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
    dk = torch.tensor(k.size(-1), dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)
    

    if mask is not None:
        mask = mask.squeeze(1)
        scaled_attention_logits += (mask * -1e9)

    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        
        self.wq = nn.Linear(d_model, d_model).to("cuda")
        self.wk = nn.Linear(d_model, d_model).to("cuda")
        self.wv = nn.Linear(d_model, d_model).to("cuda")
        self.dense = nn.Linear(d_model, d_model).to("cuda")
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)
        q = q.to("cuda")
        k = k.to("cuda")
        k = k.to("cuda")

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = scaled_attention.contiguous().view(batch_size,-1, self.d_model)  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights

    

class PointWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PointWiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
