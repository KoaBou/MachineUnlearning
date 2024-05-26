import torch
from torch.utils.data import DataLoader, Dataset


class UnlearnDataset(Dataset):
    def __init__(self, retain: DataLoader, forget: DataLoader):
        super(UnlearnDataset, self).__init__()
        self.retain = retain
        self.forget = forget
        self.forget_len = len(forget)
        self.retain_len = len(retain)
        self.len = self.retain_len + self.forget_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if index < self.forget_len:
            image = self.forget[index][0]
            label = 1
            return image, label
        else:
            image = self.retain[index][0]
            label = 0
            return image, label


