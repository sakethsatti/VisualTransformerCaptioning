import sys
import torch
from transformers import AutoTokenizer

sys.path.append('../')
from vt_captioning.vt_resnet import vt_resnet50
from transformer_code.vt_captioning import VTCaptionModel
import pandas as pd
from PIL import Image
import torchvision
from transformer_code.mha import create_look_ahead_mask, create_padding_mask

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
VOCAB_SIZE = len(tokenizer)


transform = torchvision.transforms.Compose([
     torchvision.transforms.Resize((300, 300)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

feature_extractor = vt_resnet50(
            pretrained=True,
            freeze='full_freeze', # only freezes resnet
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

model = VTCaptionModel(feature_extractor, num_layers = 8, d_model = 1024, num_heads = 16, dff = 2048, row_size = 1, col_size = 1,
                       target_vocab_size = VOCAB_SIZE, max_pos_encoding=VOCAB_SIZE, rate=0.2)

model.to(device)

model.load_state_dict(torch.load('VTResCaptioner.pt'))

dataset = pd.read_csv('cocoval.csv')

def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tar.size(1)).to(device)
    dec_target_padding_mask = create_padding_mask(tar).to(device)

    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
    return combined_mask

def evaluate(image):

    input = transform(image).unsqueeze(0).to("cuda") 
        
    decoder_input = ["[CLS]"] + ["[PAD]"] * 19
    decoder_input = tokenizer(decoder_input, add_special_tokens = False, return_tensors="pt")["input_ids"].to("cuda").permute(1,0)
    
    decoder_input = torch.cat((decoder_input, decoder_input), dim=0)
    input = torch.cat((input, input), dim=0)
    
 
    result = []  # Word list
    
    for i in range(19):
        with torch.no_grad():
            dec_mask = create_masks_decoder(decoder_input).to("cuda")
            predictions, _ = model(input, decoder_input, dec_mask)

            predicted_id = torch.argmax(predictions.permute(0,2,1), dim=-1)[0][i].item()
            
            if tokenizer.decode(predicted_id) == "[SEP]" or tokenizer.decode(predicted_id) == "[PAD]":
                return result

            result.append(tokenizer.decode(predicted_id))
        
            decoder_input[0, i+1] = predicted_id

    return result

index_random = int(input("Enter a number: "))

print("ground truth: ", dataset["captions"][index_random])

print(" ".join(evaluate(Image.open(dataset.path[index_random]))))
