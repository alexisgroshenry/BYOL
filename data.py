import torch.nn as nn
import torchvision.transforms as transforms
from image_loader import ImageFolderLoader


def pad(img, size_max=500):
    """
    Pads images to the max dimension of the image
    """
    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)
    
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    return transforms.functional.pad(
        img,
        (pad_left, pad_top, pad_right, pad_bottom),
        fill=tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406))))

input_size = 224
data_transforms_train = transforms.Compose([
    transforms.Lambda(pad),
    transforms.Resize((input_size,input_size)),
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
    transforms.Lambda(pad),
    transforms.Resize((input_size,input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])


data_transforms_online = transforms.Compose([
    transforms.Lambda(pad),
    transforms.Resize((input_size,input_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(nn.ModuleList(
        [transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.2)]),p=.8),
    transforms.GaussianBlur(23, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])


data_transforms_target = transforms.Compose([
    transforms.Lambda(pad),
    transforms.Resize((input_size,input_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(nn.ModuleList(
        [transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.2)]),p=.8),
    transforms.RandomApply(nn.ModuleList(
        [transforms.GaussianBlur(23, sigma=(0.1, 2.0))]), p=.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

