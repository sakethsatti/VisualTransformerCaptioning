import torch
from vt_captioning.vt_resnet import vt_resnet50
from transformer_code.vt_captioning import VTCaptionModel
import os

VOCAB_SIZE = 30522
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    feature_extractor = vt_resnet50(
            pretrained=True,
            freeze='full_freeze',
            tokens=16,
            token_channels=128,
            input_dim=1024,
            vt_channels=2048,
            transformer_enc_layers=2,
            transformer_heads=8,
            transformer_fc_dim=2048,
            image_channels=3,
        )
    
    model = VTCaptionModel(feature_extractor, num_layers = 8, d_model = 512, num_heads = 16, dff = 2048, row_size = 1, col_size = 1, target_vocab_size = VOCAB_SIZE,
                max_pos_encoding=VOCAB_SIZE, rate=0.2)
    
    model(torch.ones((1, 3, 300, 300)), 102, True)