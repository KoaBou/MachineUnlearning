import numpy as np
import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from copy import deepcopy

from torch.nn import functional as F

from models.models import *
from dataset.dataset import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def finetune(epoch, model: Module, loader: DataLoader) -> None:
    global DEVICE
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    running_loss = 0
    last_lost = 0

    for i, data in enumerate(loader):
        model.train()
        optimizer.zero_grad()

        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f"batch {i+1} loss: {last_loss:.4f}")
            running_loss = 0

    model.eval()

class BadTeacher():
    def __init__(self, model: Module, retain_loader: DataLoader, forget_loader: DataLoader, model_name: str = "resnet18"
                 , optimizer: str = "sgd", dataset: str = "cifar10", batch_size: int = 256) -> None:

        self.good_teachers = deepcopy(model)
        self.model = model

        if dataset == "cifar10":
            self.num_classes = 10
        elif dataset == "mufac":
            self.num_classes = 8

        if model_name == "resnet18":
            self.bad_teachers = get_resnet18()
        elif model_name == "vgg16":
            self.bad_teachers = get_vgg16()

        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters())
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters())

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.KL_temperature = 1
        self.losses = []
        self.batch_size = batch_size
        self.prepare_data(retain_loader, forget_loader)
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def fit(self, epoch) -> None:
        for i, data in self.loader:
            inputs, labels = data
            inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)
            self.optimizer.zero_grad()

            with torch.no_grad():
                good_outputs = self.good_teachers(inputs)
                bad_outputs = self.bad_teachers(inputs)
            outputs = self.model(inputs)
            loss = self.cal_loss(outputs, labels, good_outputs, bad_outputs)
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())

    def cal_loss(self, outputs, labels, good_outputs, bad_outputs):
        labels = torch.unsqueeze(labels, dim=1)

        with torch.no_grad():
            good_outputs = F.softmax(good_outputs/self.KL_temperature, dim=1)
            bad_outputs = F.softmax(bad_outputs/self.KL_temperature, dim=1)

            teacher_outputs = labels * bad_outputs + (1 - labels) * good_outputs

        student_outputs = F.log_softmax(teacher_outputs/self.KL_temperature, dim=1)
        return F.kl_div(student_outputs, teacher_outputs)

    def prepare_data(self, retain_loader: DataLoader, forget_loader: DataLoader):
        images = []
        for image, _ in retain_loader:
            images.append(image)
        for image, _ in forget_loader:
            images.append(image)

        images = np.array(images)
        labels = np.concatenate([np.zeros((len(retain_loader))) + np.ones((len(forget_loader)))])

        images = torch.Tensor(images)
        labels = torch.Tensor(labels)

        self.dataset = torch.utils.data.TensorDataset(images, labels)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


class FAT():
    def __init__(self, model: Module, retain_loader: DataLoader, forget_loader: DataLoader,
                 model_name: str = "resnet18", optimizer: str = "sgd", KL_temperature: int = 1,
                 batch_size: int = 256):
        self.good = deepcopy(model)
        self.model = model

        self.model_name = model_name

        if self.model_name == "resnet18":
            self.bad = get_resnet18()
        elif self.model_name == "vgg16":
            self.bad = get_vgg16()

        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.batch_size = batch_size
        dataset = UnlearnDataset(self.retain_loader, self.forget_loader)
        self.unlearn_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters())
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters())

        self.criterion = nn.CrossEntropyLoss()
        self.KL_temperature = KL_temperature
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.losses = {}

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def badtrain(self, epochs: int):
        for epoch in range(1, epochs+1):
            self.fit("bad", self.retain_loader)

    def finetune(self, model_name: str, epochs: int):

        for epoch in range(1, epochs+1):
            self.fit(model_name, self.retain_loader, "finetune")

    def unlearn(self, epochs):
        self.losses["unlearn"] = self.losses.get("unlearn", [])

        for epoch in range(1, epochs+1):
            pass

    def unlearn_one_epoch(self):
        loss = self.losses["unlearn"]

        input

    def fit(self, model_name: str, loader: DataLoader, train_type: str = "train"):
        if model_name == "bad":
            model = self.bad
        elif model_name == "good":
            model = self.good
        else:
            model = self.model

        loss_name = f"{model_name}_{train_type}_loss"
        self.losses[loss_name] = self.losses.get(loss_name, [])
        losses = self.losses[loss_name]

        for i, data in enumerate(loader):

            inputs, labels = data
            inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)

            self.optimizer.zero_grad()

            outputs = model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())





if __name__ == "__main__":
    # from models import models
    # import torchvision
    #
    # dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
    # loader = torch.utils.data.DataLoader(dataset, batch_size=256)
    # model = model.get_resnet18(10)
    #
    # finetune(epoch=1, model=model, loader=loader)
