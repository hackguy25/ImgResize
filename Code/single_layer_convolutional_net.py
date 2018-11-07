#In this file a single layer convolutional neural network is described, along
#with a couple of helper functions that accompany it.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, size_conv_kernel = 25):

        super(Net, self).__init__()
        
        #definition of the convolution size
        #size_conv_kernel - number of pixels in x and y dimensions taken as input
        self.conv = nn.Conv2d(3, 3, size_conv_kernel)

    def forward(self, x):

        #definition of the forward pass of the net
        x = self.conv(x)
        #clamping function acts both as an activation function and an adjustment
        #that lets the program save the picture properly
        x = torch.clamp(x, 0, 255)
        return x

def createNet (device, learnRate = 0.001, kernel = 25):

    #function that creates a new instance of the net with random parameters
    net = Net(size_conv_kernel = kernel).to(device)
    opt = torch.optim.Adam(net.parameters(), lr = learnRate)
    crit = nn.SmoothL1Loss().to(device)
    net.eval()
    return net, opt, crit

def loadNet (path, device, learnRate = 0.001, kernel = 25):

    #function that creates a new instance of the net and loads saved parameters
    #from given path
    net = Net(size_conv_kernel = kernel).to(device)
    net.load_state_dict(torch.load(path + "netStateDict.ph"))
    opt = torch.optim.Adam(net.parameters(), lr = learnRate)
    opt.load_state_dict(torch.load(path + "optStateDict.ph"))
    crit = nn.SmoothL1Loss().to(device)
    net.eval()
    return net, opt, crit

def saveNet (net, opt, pathFolder):
    
    #function that save the current net to path
    torch.save(net.state_dict(), pathFolder + "netStateDict.ph")
    torch.save(opt.state_dict(), pathFolder + "optStateDict.ph")

def train (net, opt, crit, input, target):

    #function that takes an input and target tensors and performs a single training
    #step on the net with them
    opt.zero_grad()
    output = net(input)
    loss = crit(output, target)
    loss.backward()
    opt.step()
    #returns the norm of the loss of the current training step
    return loss.norm().item()

def tensorToNumpy (tensor):
    
    #function that converts a pyTorch Tensor to a numpy ndarray
    return tensor.detach().cpu().numpy()
