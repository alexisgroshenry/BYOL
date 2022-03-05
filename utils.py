import torch
import torch.optim as optim

from model import ResNet

def init_model(args, num_classes=20, mode='classif'):
    '''
    Initialize model architecture
    '''
    return ResNet(args.resnet_size, pretrained=args.pretrained, train_last_layer=args.train_last_layer, mode=mode, num_classes=num_classes)


def load_weights(args, model):
    '''
    Handle the different cases of loading weights from a checkpoint
    '''
    state_dict = torch.load(args.weights)

    if args.byol:
        # drop the projector and predictor layers from BYOL
        for k in state_dict.keys():
            if k[:2]=='fc':
                state_dict.pop(k)
        # replace them with a layer for the classification
        for k in ["fc.weight", "fc.bias"]:
            state_dict[k] = model.state_dict()[k]
        
    model.load_state_dict(state_dict)


def init_optim(args,model):
    '''
    Initialize optimizer
    '''
    return optim.Adam(params=model.parameters(), lr=args.lr)


def init_lr_scheduler(optimizer):
    '''
    Initialize the learning rate scheduler
    '''
    warmup_epochs = 5
    warmup_lr = lambda epoch: 1
    train_lr = lambda epoch: 0.5 * 0.95 ** (epoch - warmup_epochs)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    train_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=train_lr)
    return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, train_scheduler], milestones=[warmup_epochs])


def save_list_to_file(path, thelist):
    '''
    Tool function to save a list to a .txt file
    '''
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)