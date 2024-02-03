import torch
import torchvision
import pandas as pd
import torch.nn.functional as F
from vt_captioning.vt_resnet import vt_resnet50
from transformers import AutoTokenizer
from transformer_code.vt_captioning import VTCaptionModel
from transformer_code.mha import create_look_ahead_mask, create_padding_mask
from vizwiz import VizWiz
import json
import time

VOCAB_SIZE = 30522
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 60

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

transform = torchvision.transforms.Compose([
     torchvision.transforms.Resize((300, 300)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

VizWiz_train = VizWiz(train_data.image_file.to_list(), train_data.captions.to_list(), transform, tokenizer)
VizWiz_test = VizWiz(test_data.image_file.to_list(), test_data.captions.to_list(), transform, tokenizer)

train_dataloader = torch.utils.data.DataLoader(VizWiz_train, batch_size=64, shuffle=True, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(VizWiz_test, batch_size=64, shuffle=True, drop_last=True)

def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tar.size(1)).to(device)
    dec_target_padding_mask = create_padding_mask(tar).to(device)

    combined_mask = torch.max(dec_target_padding_mask.unsqueeze(1), look_ahead_mask)
    return combined_mask

def train_step(img_tensor, tar, transformer, optimizer, train_loss, train_accuracy):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    dec_mask = create_masks_decoder(tar_inp)

    
    optimizer.zero_grad()
    print(img_tensor.shape, tar_inp.shape, dec_mask.shape)
    predictions, _ = transformer(img_tensor, tar_inp, dec_mask)


    loss = F.cross_entropy(predictions, tar_real)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    train_accuracy += torch.sum(torch.argmax(predictions.permute(0, 2, 1), dim=-1) == tar_real).item() / (tar_real.size(1) * tar_real.size(0))

    return train_loss, train_accuracy

def evaluate(model, val_loader):
    val_loss = 0.0
    val_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, (img_tensor, tar) in enumerate(val_loader):
            img_tensor = img_tensor.to(device)  # Move tensors to GPU if available
            tar = tar.to(device)

            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            dec_mask = create_masks_decoder(tar_inp)

            predictions, _ = model(img_tensor, tar_inp, dec_mask)
            loss = F.cross_entropy(predictions, tar_real)

            val_loss += loss.item()
            val_accuracy += torch.sum(torch.argmax(predictions.permute(0, 2, 1), dim=-1) == tar_real).item() / (tar_real.size(1) * tar_real.size(0))

    return val_loss / len(val_loader), 100.0 * val_accuracy / len(val_loader)

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
    
    feature_extractor.to(device)

    model = VTCaptionModel(feature_extractor, num_layers = 8, d_model = 512, num_heads = 16, dff = 2048, row_size = 1, col_size = 1, target_vocab_size = VOCAB_SIZE,
                max_pos_encoding=VOCAB_SIZE, rate=0.2)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    full_history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        print("Epoch ", epoch + 1)
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        start_time = time.time()

        for batch_idx, (img_tensor, tar) in enumerate(train_dataloader):
            optimizer.zero_grad()
            img_tensor = img_tensor.to(device)
            tar = tar.to(device)
            train_loss, train_accuracy = train_step(img_tensor, tar, model, optimizer, train_loss, train_accuracy)
            
            if batch_idx % 50 == 0 or batch_idx == len(train_dataloader) - 1:
                print(f"Batch [{batch_idx}/{len(train_dataloader)}] ")
        
        

        scheduler.step()  # Adjust learning rate
        test_loss, test_accuracy = evaluate(model, test_dataloader)

        print()
        print(f"Train Loss: {train_loss / len(train_dataloader):.4f} \n"
            f"Train Accuracy: {100.0 * train_accuracy / len(train_dataloader):.2f}% \n"
            f"Test Loss: {test_loss:.4f} \n"
            f"Test Accuracy: {test_accuracy:.2f}% \n"
            f"Time taken: {time.time() - start_time:.2f} seconds"
            )
        
        print()
        print()
        
        full_history['train_loss'].append(train_loss / len(train_dataloader))
        full_history['train_accuracy'].append(100.0 * train_accuracy / len(train_dataloader))
        full_history['val_loss'].append(test_loss)
        full_history['val_accuracy'].append(test_accuracy)

        with open('full_history.json', 'w') as fp:
            json.dump(full_history, fp)

        torch.save(model.state_dict(), 'VTResCaptioner.pt')