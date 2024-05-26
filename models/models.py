import torchvision
import torch


def get_resnet18(num_classes=10):
    model = torchvision.models.resnet18()
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    return model
def get_vgg16(num_classes=10):
    model = torchvision.models.vgg16()
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)

    return model

if __name__ == '__main__':
    model = get_vgg16()
    print(model)