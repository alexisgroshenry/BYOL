from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

# define projector architecture following BYOL article
class Proj_MLP(nn.Module):
    def __init__(self):
        super(Proj_MLP, self).__init__()
        self.fc1 = nn.Linear(2048, 4096)
        self.bn = nn.BatchNorm1d(num_features=4096)
        self.fc2 = nn.Linear(4096, 256)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        return self.fc2(x)


# define predictor architecture following BYOL article
class Pred_MLP(nn.Module):
    def __init__(self):
        super(Pred_MLP, self).__init__()
        self.fc1 = nn.Linear(256, 4096)
        self.bn = nn.BatchNorm1d(num_features=4096)
        self.fc2 = nn.Linear(4096, 256)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        return self.fc2(x)


def set_parameter_requires_grad(model, train_last_layer = False):
    # freeze the weights of the convolutional layers
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze the weights of the last convolutional layers
    if train_last_layer:
        for param in model.layer4.parameters():
            param.requires_grad = True


def ResNet(size=152, pretrained=True, train_last_layer=True, mode='classif', num_classes=20):
    if size==50:
        model = models.resnet50(pretrained)
    elif size==152:
        model = models.resnet152(pretrained)

    set_parameter_requires_grad(model, train_last_layer)

    # model used for classification
    if mode=='classif':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    # model used as online model in BYOL
    elif mode=='online':
        projector = Proj_MLP()
        predictor = Pred_MLP()
        model.fc = nn.Sequential(
            projector,
            predictor
        )

    # model used as target model in BYOL
    elif mode=='target':
        projector = Proj_MLP()
        model.fc = nn.Sequential(
            projector
        )

    return model

