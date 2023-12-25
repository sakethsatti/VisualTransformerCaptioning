import torch
import torchvision
from torch.utils.data import random_split
import torch.nn.functional as F
import time
from vt_captioning.vt_resnet import vt_resnet50
import json
import os

BATCH_SIZE_TRAIN = 15
BATCH_SIZE_TEST = 15
TRAIN_RATIO = 0.8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DL_PATH = "../SUN397/" # add your own file path
N_EPOCHS = 50
MODEL_PATH = "ViTRes.pt"
LR = 0.01

transform = torchvision.transforms.Compose([
     torchvision.transforms.Resize((350, 350)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # Works well for Imagenet

dataset = torchvision.datasets.SUN397(DL_PATH, transform=transform) # add argument "download = true" if not download

train_size = int(TRAIN_RATIO * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN,
                                          shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST,
                                         shuffle=False, drop_last=True)

def train(model, optimizer, data_loader, loss_history):
   
    total_samples = len(data_loader.dataset)
    
    model.train()
    model.to(DEVICE)

    for i, (data, target) in enumerate(data_loader):
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + 
                  '/' + '{:5}'.format(total_samples) +
                  ' (' +
                  '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))

def evaluate(model, data_loader, loss_history):

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():

        for data, target in data_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('Loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')
    accuracy = 100.0 * correct_samples / total_samples
    return accuracy


if __name__ == "__main__":
    model = vt_resnet50(
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
            num_classes=397,
        )
    model = model.to(DEVICE)

    if os.path.exists("ViTRes.pt"):
        model.load_state_dict(torch.load("ViTRes.pt"))
        print("Model loaded from ViTRes.py")

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=.9,weight_decay=4e-5, nesterov=True)

    train_loss_history, test_loss_history, accuracy_history = [], [], []
    if os.path.exists("loss_history.json"):
        loss_history = json.load(open("loss_history.json"))
        train_loss_history = loss_history["train_loss_history"]
        test_loss_history = loss_history["test_loss_history"]
        accuracy_history = loss_history["accuracy_history"]
    else:
        train_loss_history, test_loss_history, accuracy_history = [], [], []

    for epoch in range(11, N_EPOCHS + 1):
        print('Epoch:', epoch)
        start_time = time.time()
        train(model, optimizer, train_loader, train_loss_history)
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        accuracy = evaluate(model, test_loader, test_loss_history)
        accuracy_history.append(accuracy.item())

        if accuracy_history[-1] < (sum(accuracy_history[-7:1])/3):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print('New LR rate: ', optimizer.param_groups[0]['lr'])
        
        torch.save(model.state_dict(), MODEL_PATH)
        json.dump({"train_loss_history": train_loss_history, "test_loss_history": test_loss_history, "accuracy_history": accuracy_history}, open("loss_history.json", 'w'))    
