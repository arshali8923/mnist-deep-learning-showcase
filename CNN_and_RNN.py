import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets , transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5) , (0.5))
])
train_dataset = datasets.MNIST(root = "./data" , train = True , download = True , transform = transform)
test_dataset = datasets.MNIST(root = "./data" , train = False , download = True , transform = transform)
# print(train_dataset)

train_loader = DataLoader(train_dataset , batch_size = 64 , shuffle = True)
test_loader = DataLoader(test_dataset , batch_size = 64)

"""CNN"""
class CNN(nn.Module):
    def __init__(self):
        super(CNN , self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1 , 32 , kernel_size = 3 , padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32  , 64 , kernel_size = 3 , padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64 , 128 , kernel_size = 3 , padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)

        )
        self.fc_layers = nn.Sequential(
            nn.Linear(3*3*128 , 256),
            nn.ReLU(),

            nn.Linear(256 , 10)
        )

    def forward(self , x):
        x = self.conv_layers(x)
        x = x.view(x.size(0) , -1)
        x = self.fc_layers(x)

        return x

cnn_model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters())

"""training"""
epochs = 5
for epoch in range(epochs):
    epoch_train_loss = 0.0

    for images , labels in train_loader:
        optimizer.zero_grad()

        outputs = cnn_model(images)
        loss = criterion(outputs , labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    # print(f"epoch is {epoch + 1} / {epochs} and loss is {epoch_train_loss / len(train_loader)}")

"""evaluation"""
cnn_model.eval()
correct_vals = 0
total_vals = 0

with torch.no_grad():
    for images , labels in test_loader:
        outputs = cnn_model(images)
        _ , predicted = torch.max(outputs , 1)
        correct_vals += (predicted == labels).sum().item()
        total_vals += labels.size(0)

# print(f"CNN accuracy is {correct_vals / total_vals * 100}%")

"""RNN"""
class RNN(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1):
        super(RNN, self).__init__()
        self.rnn_layers = nn.LSTM(input_size=28, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.squeeze(1)   # [batch, 28, 28]
        _, (hn, _) = self.rnn_layers(x)
        return self.fc_layers(hn[-1])

rnn_model = RNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn_model.parameters())

"""training"""
epochs = 5
for epoch in range(epochs):
    epoch_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = rnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    # print(f"Epoch {epoch+1}/{epochs}, Loss = {epoch_loss/len(train_loader)}")

"""Eval"""
correct = 0
total = 0
rnn_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = rnn_model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# print(f"RNN Accuracy = {correct/total*100}%")

"""confusion matrix"""
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(model, loader, model_name="Model"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Classification Report
    print(f"\n{model_name} Classification Report:\n")
    print(classification_report(all_labels, all_preds, digits=4))

plot_confusion_matrix(cnn_model, test_loader, "CNN")
plot_confusion_matrix(rnn_model, test_loader, "RNN")

