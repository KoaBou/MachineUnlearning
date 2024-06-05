import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random

def get_class_unlearn_set(dataset: Dataset, unlearn_class: int):
    unlearn_image = []
    unlearn_label = []
    retain_image = []
    retain_label = []
    for data in dataset:
        if data[1] == unlearn_class:
            unlearn_image.append(data[0])
            unlearn_label.append(data[1])
        else:
            retain_image.append(data[0])
            retain_label.append(data[1])

    unlearn_set = CustomDataset(unlearn_image, unlearn_label)
    retain_set = CustomDataset(retain_image, retain_label)

    return unlearn_set, retain_set


def get_random_unlearn_set(dataset: Dataset, num_sample: int):
    retain_set, unlearn_set = torch.utils.data.random_split(dataset, [len(dataset)-num_sample, num_sample])
    return unlearn_set, retain_set


class CustomDataset(Dataset):
    def __init__(self, images: list, labels: list):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class UnlearnDataset(Dataset):
    def __init__(self, retain: Dataset, forget: Dataset):
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

class AmnesiacDataset(Dataset):
    def __init__(self, retain: Dataset, forget: Dataset, dataset: str = "cifar10"):
        super(AmnesiacDataset, self).__init__()
        self.retain = retain
        self.forget = forget
        self.retain_len = len(retain)
        self.forget_len = len(forget)
        self.len = self.retain_len + self.forget_len

        if dataset == "cifar10":
            self.num_classes = 10
        elif dataset == "MUFAC":
            self.num_classes = 8

        self.classesList = list(range(0, self.num_classes))

    def __len__(self):
        return len(self)

    def __getitem__(self, index):
        if index < self.forget_len:
            image = self.forget[index][0]
            label = random.choice(self.classesList)
        else:
            image = self.retain[index][0]
            label = self.retain[index][1]

        return image, label


if __name__ == '__main__':
    import torchvision
    dataset = torchvision.datasets.CIFAR10(root='../data', train=True, transform=torchvision.transforms.ToTensor())

    unlearn_set, retain_set = get_random_unlearn_set(dataset, 5)

    amnesiac_set = AmnesiacDataset(retain_set, unlearn_set)

    for i in range(10):
        print(amnesiac_set[i][1])
