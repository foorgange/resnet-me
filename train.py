import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_loaders
from model import ResNet18
from evaluate import evaluate
import os

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    total = 0
    correct = 0

    for inputs, targets in tqdm(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, correct / total

def train(root_dir, epochs=20, batch_size=64, lr=0.001, save_path='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_loader, test_loader = get_loaders(root_dir, batch_size)
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch}/{epochs} Train loss: {train_loss:.4f} Train acc: {train_acc:.4f}')

        test_acc, test_f1 = evaluate(model, test_loader, device)
        print(f'Validation Acc: {test_acc:.4f}, F1-score: {test_f1:.4f}')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print('Model saved!')

if __name__ == '__main__':
    root_dir = './datas/cifar-10-python'
    train(root_dir)
