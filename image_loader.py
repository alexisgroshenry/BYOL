import torch
import torch.utils.data as data_utils
from torchvision.datasets.cifar import CIFAR10
from PIL import Image

def find_classes(classes):
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def create_dataloaders(batch_size=10, train_transform=None, test_transform=None):
    # load datasets
    train_cifar = CIFAR10('cifar', train=True, download=True, transform=train_transform)
    test_set = CIFAR10('cifar', train=False, download=True, transform=test_transform)

    # define indices of labeled and unlabeled training images
    label_indices = torch.arange(100)
    
    # create the corresponding datasets
    train_set = data_utils.Subset(train_cifar, label_indices)

    # create the dataaloaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=1)
    return train_loader, test_loader



class ImageFolderLoader(data_utils.Dataset):
    '''
    Define customized class to load the same image with two different transformations
    '''
    def __init__(self, transform_1=None, transform_2=None):
        cifar = CIFAR10('cifar', train=True, download=True, transform=None)
        classes, class_to_idx = find_classes(cifar.classes)
        
        self.imgs = cifar.data
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def __getitem__(self, index):
        img = Image.fromarray(self.imgs[index])
        if self.transform_1 is not None:
            img1 = self.transform_1(img)
        if self.transform_2 is not None:
            img2 = self.transform_2(img)

        return img1, img2

    def __len__(self):
        return len(self.imgs)