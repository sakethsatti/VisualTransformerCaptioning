import torch
import torchvision
from torch.utils.data import random_split
import torch.nn.functional as F
import time
from visual_transformer_code.vt import ViTResNet
from visual_transformer_code.basicblock import BasicBlock
import json

BATCH_SIZE_TRAIN = 150
BATCH_SIZE_TEST = 150
TRAIN_RATIO = 0.8
DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
DL_PATH = "../SUN397/" # add your own file path
N_EPOCHS = 150
MODEL_PATH = "ViTRes.pt"

transform = torchvision.transforms.Compose([
     torchvision.transforms.Resize((350, 350)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # Works well for Imagenet

dataset = torchvision.datasets.SUN397(DL_PATH, transform=transform) # add argument "download = true" if not download

train_size = int(TRAIN_RATIO * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST,
                                         shuffle=False)

def train(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        data = data.to(DEVICE_NAME)
        target = target.to(DEVICE_NAME)
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())

def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(DEVICE_NAME)
            target = target.to(DEVICE_NAME)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')


if __name__ == "__main__":
    model = ViTResNet(BasicBlock, [3, 3, 3])
    model = model.to(DEVICE_NAME)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9,weight_decay=1e-4)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[35,48],gamma = 0.1)

    train_loss_history, test_loss_history = [], []

    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        start_time = time.time()
        train(model, optimizer, train_loader, train_loss_history)
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        evaluate(model, test_loader, test_loss_history)

    print('Execution time')
    
    torch.save(model.state_dict(), MODEL_PATH)
    json.dump({"train_loss_history": train_loss_history, "test_loss_history": test_loss_history}, open("loss_history.json", 'w'))