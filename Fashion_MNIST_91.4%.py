import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import FashionMNIST

import pandas as pd
import numpy as np

learning_rate = 0.001
batch_size = 100
num_classes = 10
epochs = 8

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5))
])

download_root = './MNIST_DATASET'

train_dataset = FashionMNIST(download_root, transform=mnist_transform, train=True, download=True)
test_dataset = FashionMNIST(download_root, transform=mnist_transform, train=False, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.batch2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 6 * 6, 600)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)
        
        
    def forward(self, input):
        output = self.pool(F.relu(self.batch1(self.conv1(input))))
        output = self.pool(F.relu(self.batch2(self.conv2(output))))
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

model = ConvNet()
critertion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)

for epoch in range(epochs):
    total = 0
    correct = 0
    for i, data in enumerate(train_loader):
        images, labels = data
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = critertion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, epochs, i+1, total_step, loss.item()))

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy : {0:.2f}%' .format(correct / total * 100))

PATH = './FashionMNIST_2st.pth'
torch.save(model.state_dict(), PATH)

# 1차
# epoch: 3, dropout: 0.25
# accuracy : 90%

# 2차
# epoch: 8, dropout: 0.25
# accuracy : 91.4%

# 3차
# epoch: 12, dropout: 0.35
# accuracy : 90.63%

# 4차
# epoch: 12, dropout: 0.25
# accuracy : 90.4%

# 5차
# epoch: 15, dropout: 0.5
# accuracy : 90.2%

# 6차
# epoch 10, dropout 0.25 (conv층 1, fc층 1)
# accuracy : 