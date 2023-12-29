import torch
import torchvision
import pandas as pd
import torch.nn.functional as F
from vt_captioning.vt_resnet import vt_resnet50
from transformers import AutoTokenizer
from transformer_code.vt_captioning import VTCaptionModel
from transformer_code.mha import create_look_ahead_mask, create_padding_mask
from vizwiz import VizWiz

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
    look_ahead_mask = create_look_ahead_mask(tar.size(1))
    dec_target_padding_mask = create_padding_mask(tar)

    combined_mask = torch.max(dec_target_padding_mask.unsqueeze(1), look_ahead_mask)
    return combined_mask

def train_step(img_tensor, tar, transformer, optimizer, train_loss, train_accuracy):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    dec_mask = create_masks_decoder(tar_inp)

    optimizer.zero_grad()
    predictions, _ = transformer(img_tensor, tar_inp, True, dec_mask)
    loss = F.cross_entropy(tar_real, predictions)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    train_accuracy += torch.sum(torch.argmax(predictions, dim=-1) == tar_real).item() / tar_real.size(1)

    return train_loss, train_accuracy

def evaluate(model, val_loader):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, (img_tensor, tar) in enumerate(val_loader):
            img_tensor = img_tensor.to(device)  # Move tensors to GPU if available
            tar = tar.to(device)

            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            dec_mask = create_masks_decoder(tar_inp)

            predictions, _ = model(img_tensor, tar_inp, False, dec_mask)
            loss = F.cross_entropy(tar_real, predictions)

            val_loss += loss.item()
            val_accuracy += torch.sum(torch.argmax(predictions, dim=-1) == tar_real).item() / tar_real.size(1)

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
    
    model = VTCaptionModel(feature_extractor, num_layers = 8, d_model = 512, num_heads = 16, dff = 2048, row_size = 1, col_size = 1, target_vocab_size = VOCAB_SIZE,
                max_pos_encoding=VOCAB_SIZE, rate=0.2)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0

        for batch_idx, (img_tensor, tar) in enumerate(train_dataloader):
            optimizer.zero_grad()
            img_tensor = img_tensor.to(device)
            tar = tar.to(device)

            train_loss, train_accuracy = train_step(img_tensor, tar, model, optimizer, train_loss, train_accuracy)
            exit()

        scheduler.step()  # Adjust learning rate
        test_loss, test_accuracy = evaluate(model, test_dataloader)


        print(f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss / len(train_dataloader):.4f} "
            f"Train Accuracy: {100.0 * train_accuracy / len(train_dataloader):.2f}% "
            f"Test Loss: {test_loss} "
            f"Test Accuracy: {test_accuracy}%"
            )