# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions

        self.c1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.r1 = nn.ReLU()
        self.m1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.r2 = nn.ReLU()
        self.m2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        self.f1 = nn.Linear(400, 256)
        self.r3 = nn.ReLU()
        self.f2 = nn.Linear(256, 128)
        self.r4 = nn.ReLU()
        self.f3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        shape_dict = {}
        # certain operations
        x = self.c1(x)
        x = self.r1(x)
        x = self.m1(x)
        shape_dict[1] = list(x.shape)
        x = self.c2(x)
        x = self.r2(x)
        x = self.m2(x)
        shape_dict[2] = list(x.shape)

        x = self.flatten(x)

        shape_dict[3] = list(x.shape)
        x = self.f1(x)
        x = self.r3(x)
        shape_dict[4] = list(x.shape)
        x = self.f2(x)
        x = self.r4(x)
        shape_dict[5] = list(x.shape)
        x = self.f3(x)
        shape_dict[6] = list(x.shape)
        out = x
		
        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0
    for i, mparams in model.named_parameters():
        if mparams.requires_grad:
            model_params += mparams.numel()
    return model_params / 1e6


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
