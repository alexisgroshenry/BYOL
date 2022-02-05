import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils import init_model, init_optim, save_list_to_file
from data import data_transforms_online, data_transforms_target, ImageFolderLoader


# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=4096, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--resnet_size', type=int, default=152, metavar='S',
                    help='size of the resnet model in [18, 50, 152] (default: 152)')
parser.add_argument('--pretrained', type=bool, default=True, metavar='PTR',
                    help='whether to use pretrained weights on ImageNet (default: True)')
parser.add_argument('--train_last_layer', type=bool, default=True, metavar='TLL',
                    help='whether to train the lasst convolutional layers of the network (default: True)')
parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)


online_model = init_model(args, mode='online')
target_model = init_model(args, mode='target')
if use_cuda:
    print('Using GPU')
    online_model.cuda()
    target_model.cuda()
else:
    print('Using CPU')

train_loss = []
# optimizer for the weights of the online network only
optimizer = init_optim(args, online_model)
tau0 = 0.996

image_datasets = ImageFolderLoader(args.data, data_transforms_online, data_transforms_target)
train_loader = torch.utils.data.DataLoader(
        image_datasets, batch_size=args.batch_size,
        shuffle=True, num_workers=0)

def train(epoch):
    total_loss = 0.
    online_model.train()
    target_model.eval()
    optimizer.zero_grad()
    
    for batch_idx, (x1, x2) in enumerate(tqdm(train_loader)):
        if use_cuda:
            x1 = x1.cuda()
            x2 = x2.cuda()
        online_output_1 = online_model(x1)
        online_output_2 = online_model(x2)
        target_output_1 = target_model(x1)
        target_output_2 = target_model(x2)

        # implementing MSE for normalized input vectors
        def regression_loss(x, y):
            norm_x, norm_y = torch.norm(x, dim=1), torch.norm(y, dim=1)
            return 2. - 2. * torch.mean(torch.sum(x * y, dim=1) / (norm_x * norm_y))

        # compute loss
        loss = regression_loss(online_output_1, target_output_2)
        loss += regression_loss(online_output_2, target_output_1)
        total_loss += loss.cpu().item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        # update the weights of the target network
        weights_online = online_model.state_dict()
        weights_target = target_model.state_dict()
        tau = 1 - (1 - tau0) * (np.cos(np.pi * epoch * batch_idx / (args.epochs * len(train_loader)) ) + 1) / 2
        new_weights = {k: tau * weights_target[k] + (1 - tau) * weights_online[k] for k in weights_target.keys()}
        target_model.load_state_dict(new_weights)

    train_loss.append(total_loss)
    print('Train Epoch {}: Average train loss: {:.4f}'.format(epoch, total_loss))



warmup_epochs = 10
scaled_lr = args.batch_size / 256.

warmup_lr = lambda epoch: scaled_lr * (epoch+1) / warmup_epochs
train_lr = lambda epoch: scaled_lr * (np.cos(np.pi * ( epoch - warmup_epochs) / args.epochs) + 1) / 2
warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
train_scheduler  = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=train_lr)
lr_scheduler     = optim.lr_scheduler.SequentialLR(optimizer,
                        schedulers=[warmup_scheduler, train_scheduler],
                        milestones=[warmup_epochs]
                    )

for epoch in range(1, args.epochs + 1):
    train(epoch)
    if epoch==1:
        best_loss = train_loss[-1]
    if epoch==1 or train_loss[-1] < best_loss:
        print('New best_loss, saving model')
        model_file = args.experiment + '/best_model.pth'
        torch.save(online_model.state_dict(), model_file)

print('Saving last model')
model_file = args.experiment + '/last_model.pth'
torch.save(online_model.state_dict(), model_file)

for name in ['train_loss']:
    save_list_to_file(args.experiment + '/' + name + '.txt', locals()[name])