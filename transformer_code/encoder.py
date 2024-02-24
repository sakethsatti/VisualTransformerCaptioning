import torch
from torch import nn
from transformer_code.positional_encoding import positional_encoding_2d
from transformer_code.encoder_layer import EncoderLayer


class Encoder(nn.Module):
   def __init__(self, num_layers, embed_size, d_model, num_heads, dff, row_size,col_size,rate=0.1):
      super(Encoder, self).__init__()
      self.d_model = d_model
      self.num_layers = num_layers

      
      self.embedding = nn.Linear(embed_size, self.d_model)
      self.linear_activation = nn.LeakyReLU(0.01)

      self.pos_encoding = positional_encoding_2d(row_size,col_size,self.d_model)

      self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
      self.dropout = nn.Dropout(rate)

   def forward(self, x, mask=None):
      seq_len = x.size(1)
      x = self.embedding(x)  # (batch_size, input_seq_len(H*W), d_model)
      x = self.linear_activation(x)
      self.pos_encoding = self.pos_encoding.to(x.device)
      x += self.pos_encoding[:, :seq_len, :]
      x = self.dropout(x)
      
      for i in range(self.num_layers):
         x = self.enc_layers[i](x, mask)
      return x # (batch_size, input_seq_len, d_model)
