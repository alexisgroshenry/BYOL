import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from utils import init_model, load_weights
from data import default_transform, augmented_transform
from image_loader import create_dataloaders


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=20, metavar='B',
                    help='input batch size for training (default: 20)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--resnet_size', type=int, default=101, metavar='S',
                    help='size of the resnet model (default: 101)')
parser.add_argument('--pretrained', type=bool, default=False, metavar='PTR',
                    help='whether to use pretrained weights on ImageNet (default: False)')
parser.add_argument('--freeze', type=bool, default=True, metavar='FRZ',
                    help='whether to freeze the weights (default: True)')
parser.add_argument('--train_last_layer', type=bool, default=False, metavar='TLL',
                    help='whether to train the last convolutional layers of the network (default: False)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs from BYOL are located.')



def train(model, train_loader, test_loader, num_epochs=10, lr=1e-3, test_every=5):

    # get the weights of each class in the training set to balance the loss
    train_targets = train_loader.dataset.dataset.targets[:100]
    train_count = torch.bincount(torch.Tensor(train_targets).int())
    train_weights = train_count.max()/train_count

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training on {}'.format(device))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=train_weights.to(device))
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(1,1+num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Epoch: {} Loss: {:.3f} Acc: {:.3f}% ({}/{})'.format(epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Test phase every `test_every` epochs
        if not epoch%test_every:
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            print('       Test : Loss: {:.3f} Acc: {:.3f}% ({}/{})'.format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        scheduler.step()


if __name__ == '__main__':

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    train_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        train_transform=augmented_transform,
        test_transform=default_transform
    )

    num_classes = len(train_loader.dataset.dataset.classes)
    model = init_model(args, num_classes, mode='classif')
    load_weights(args.experiment+'/best_model_103.pth', model)

    train(
        model,
        train_loader,
        test_loader,
        num_epochs=args.epochs,
        lr=args.lr
    )