import torch
from torch import nn
from transformer_code.positional_encoding import positional_encoding_1d
from transformer_code.decoder_layer import DecoderLayer

class Decoder(nn.Module):
   def __init__(self, num_layers,d_model,num_heads,dff, target_vocab_size, maximum_position_encoding,   rate=0.1):
      super(Decoder, self).__init__()
      self.d_model = d_model
      self.num_layers = num_layers

      self.embedding = nn.Embedding(target_vocab_size, d_model).to("cuda")
      self.pos_encoding = positional_encoding_1d(maximum_position_encoding, d_model).to("cuda")

      self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                         for _ in range(num_layers)]
      self.dropout = nn.Dropout(rate)

   def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
      seq_len = x.size()[1]
      attention_weights = {}
      
      x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
      x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
      x += self.pos_encoding[:, :seq_len, :]
      x = self.dropout(x)

      for i in range(self.num_layers):
         x, block1, block2 = self.dec_layers[i](x, enc_output,
                                            look_ahead_mask, padding_mask)
         
         attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
         attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
      return x, attention_weights
