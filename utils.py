import torch
import torch.optim as optim

from model import ResNet

def init_model(args, num_classes=20, mode='classif'):
    '''
    Initialize model architecture
    '''
    return ResNet(args.resnet_size, pretrained=args.pretrained, train_last_layer=args.train_last_layer, mode=mode, num_classes=num_classes)


def load_weights(path, model):
    '''
    Load weights from the representation learnt by BYOL and drop weights not used for the classification
    '''
    state_dict = torch.load(path)
    tmp_dict = state_dict.copy()
    # drop the projector and predictor layers from BYOL
    for k in state_dict.keys():
        if k[:2]=='fc':
            tmp_dict.pop(k)
    state_dict = tmp_dict
    # replace them with a layer for the classification
    for k in ["fc.weight", "fc.bias"]:
        state_dict[k] = model.state_dict()[k]
        
    model.load_state_dict(state_dict)


def init_optim(args,model):
    '''
    Initialize optimizer
    '''
    return optim.Adam(params=model.parameters(), lr=args.lr)


def save_list_to_file(path, thelist):
    '''
    Tool function to save a list to a .txt file
    '''
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)