from torch.utils.data import DataLoader
from torch.nn import Module
import torch



def accuracy(net: Module, loader: DataLoader):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    correct = 0
    total = 0

    net.to(DEVICE).eval()

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return correct / total


def fit(model, loader, optimizer, criterion, scheduler):
    pass


