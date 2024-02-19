from torch import nn
from transformer_code.encoder import Encoder
from transformer_code.decoder import Decoder

class VTCaptionModel(nn.Module):
   def __init__(self, vt_model, num_layers, d_model, num_heads, dff,row_size,col_size,
              target_vocab_size,max_pos_encoding, rate=0.1):
      super(VTCaptionModel, self).__init__()
      self.VT = vt_model
      
      self.encoder = Encoder(num_layers, self.VT.vt_channels, d_model, num_heads, dff,row_size,col_size, rate)
      self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                          target_vocab_size,max_pos_encoding, rate)
      self.final_layer = nn.Linear(d_model, target_vocab_size)

   def forward(self, img, tar, look_ahead_mask=None,dec_padding_mask=None,enc_padding_mask=None   ):
      enc_inp = self.VT(img)
      enc_output = self.encoder(enc_inp, enc_padding_mask)  # (batch_size, inp_seq_len, d_model      )
      dec_output, attention_weights = self.decoder(
      tar, enc_output, look_ahead_mask, dec_padding_mask)
      final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
      final_output = final_output.permute(0, 2, 1)
      return final_output, attention_weights
