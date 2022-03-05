import torch.nn as nn
import torchvision.transforms as transforms


input_size = 224
data_transforms_train = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(nn.ModuleList(
        [transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.2)]),p=.8),
    transforms.RandomApply(nn.ModuleList(
        [transforms.GaussianBlur(23, sigma=(0.1, 2.0))]), p=.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])


data_transforms_val = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])


data_transforms_online = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(nn.ModuleList(
        [transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.2)]),p=.8),
    transforms.GaussianBlur(23, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])


data_transforms_target = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(nn.ModuleList(
        [transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.2)]),p=.8),
    transforms.RandomApply(nn.ModuleList(
        [transforms.GaussianBlur(23, sigma=(0.1, 2.0))]), p=.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

