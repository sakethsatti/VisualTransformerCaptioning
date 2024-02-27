import torch
from vt_captioning.vt_resnet import vt_resnet50
from transformer_code.vt_captioning import VTCaptionModel
from transformer_code.mha import create_look_ahead_mask, create_padding_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB_SIZE = 30522

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
    
feature_extractor.to(device)

model = VTCaptionModel(feature_extractor, num_layers = 8, d_model = 1024, num_heads = 16, dff = 2048, row_size = 19, col_size = 19, target_vocab_size = VOCAB_SIZE, max_pos_encoding=VOCAB_SIZE, rate=0.2).to(device)

model.load_state_dict(torch.load('VTResCaptioner.pt'))

def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tar.size(1)).to(device)
    dec_target_padding_mask = create_padding_mask(tar).to(device)

    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
    return combined_mask

print("parameters:", sum(p.numel() for p in model.parameters()))
print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


predictions, _ = model(torch.ones(2, 3, 300, 300, dtype = torch.float32).to(device),
                       torch.ones(2, 20, dtype=torch.int32).to(device),
                       create_masks_decoder(torch.ones(2,20,dtype=torch.int32)).to(device))

print(predictions.size())
