import torch.utils.data as data
from torchvision.datasets.cifar import CIFAR10
from PIL import Image

def find_classes(classes):
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class ImageFolderLoader(data.Dataset):
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